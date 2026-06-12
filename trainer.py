# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
from typing import Dict, Tuple, List
from model import UniversalMicroUnit
from worlds import AbstractWorld

class MultiTaskTrainer:
    """
    A multi-task trainer designed to train a single UniversalMicroUnit across three tasks
    (SpatialShiftWorld, TemporalDelayWorld, and ContextInversionWorld) simultaneously
    to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        model: UniversalMicroUnit,
        worlds: Dict[str, AbstractWorld],
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.worlds = worlds
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Loss criteria
        self.criterion_spatial = nn.MSELoss()
        self.criterion_temporal = nn.BCEWithLogitsLoss()
        self.criterion_context = nn.BCEWithLogitsLoss()
        
        # Loss weights as specified:
        # Total_Loss = 1.0 * Spatial_Loss + 5.0 * Temporal_Loss + 5.0 * Context_Loss
        self.w_spatial = 1.0
        self.w_temporal = 5.0
        self.w_context = 5.0

        # History tracking (for training progress and plotting)
        self.history = {
            'spatial_shift': {'loss': [], 'accuracy': []},
            'temporal_delay': {'loss': [], 'accuracy': [], 'seq_accuracy': []},
            'context_inversion': {'loss': [], 'accuracy': []}
        }

    def train_epoch(self, batch_size: int, steps_per_epoch: int) -> Dict[str, float]:
        """
        Runs one epoch of training using interleaved batches and sequential gradient accumulation.
        
        Args:
            batch_size (int): Batch size for each task.
            steps_per_epoch (int): Number of steps (parameter updates) in the epoch.
            
        Returns:
            Dict[str, float]: Mean loss values computed during the training epoch.
        """
        self.model.train()
        
        epoch_losses = {
            'spatial_shift': 0.0,
            'temporal_delay': 0.0,
            'context_inversion': 0.0,
            'total': 0.0
        }
        
        for _ in range(steps_per_epoch):
            # Zero gradients for all weights before accumulation
            self.optimizer.zero_grad()
            
            # --- Task 1: SpatialShiftWorld ---
            X_s, Y_s = self.worlds['spatial_shift'].generate_batch(batch_size)
            # X_s shape: [B, 1, 5, 5], Y_s shape: [B, 2]
            pred_s = self.model(X_s, 'spatial_shift')  # shape: [B, 2]
            loss_s = self.criterion_spatial(pred_s, Y_s)
            
            # Backpropagate Spatial loss with weight 1.0
            (self.w_spatial * loss_s).backward()
            epoch_losses['spatial_shift'] += loss_s.item()
            
            # --- Task 2: TemporalDelayWorld ---
            X_t, Y_t = self.worlds['temporal_delay'].generate_batch(batch_size)
            # X_t shape: [B, 10, 1], Y_t shape: [B, 10, 1]
            pred_t = self.model(X_t, 'temporal_delay')  # shape: [B, 10, 1]
            loss_t = self.criterion_temporal(pred_t, Y_t)
            
            # Backpropagate Temporal loss with weight 5.0
            (self.w_temporal * loss_t).backward()
            epoch_losses['temporal_delay'] += loss_t.item()
            
            # --- Task 3: ContextInversionWorld ---
            X_c, Y_c = self.worlds['context_inversion'].generate_batch(batch_size)
            # X_c shape: [B, 3], Y_c shape: [B, 1]
            pred_c = self.model(X_c, 'context_inversion')  # shape: [B, 1]
            loss_c = self.criterion_context(pred_c, Y_c)
            
            # Backpropagate Context loss with weight 5.0
            (self.w_context * loss_c).backward()
            epoch_losses['context_inversion'] += loss_c.item()
            
            # Update all weights simultaneously using accumulated gradients
            self.optimizer.step()
            
            # Record total combined weighted loss
            total_loss_val = (self.w_spatial * loss_s.item() + 
                              self.w_temporal * loss_t.item() + 
                              self.w_context * loss_c.item())
            epoch_losses['total'] += total_loss_val
            
        # Average loss values over steps
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
            
        return epoch_losses

    @torch.no_grad()
    def evaluate(self, val_batch_size: int = 512) -> Dict[str, Dict[str, float]]:
        """
        Runs evaluation on a clean validation batch for each task.
        
        Args:
            val_batch_size (int): Size of the validation batch to evaluate on.
            
        Returns:
            Dict[str, Dict[str, float]]: Metrics containing loss and accuracy for all tasks.
        """
        self.model.eval()
        metrics = {}
        
        # --- Evaluate Task 1: SpatialShiftWorld ---
        X_s, Y_s = self.worlds['spatial_shift'].generate_batch(val_batch_size)
        # X_s shape: [B, 1, 5, 5], Y_s shape: [B, 2]
        pred_s = self.model(X_s, 'spatial_shift')  # shape: [B, 2]
        loss_s = self.criterion_spatial(pred_s, Y_s).item()
        
        # Absolute error < 0.5 on both x and y coordinates
        abs_err = torch.abs(pred_s - Y_s)  # shape: [B, 2]
        correct_s = (abs_err < 0.5).all(dim=-1)  # shape: [B]
        accuracy_s = correct_s.float().mean().item()
        metrics['spatial_shift'] = {'loss': loss_s, 'accuracy': accuracy_s}
        
        # --- Evaluate Task 2: TemporalDelayWorld ---
        X_t, Y_t = self.worlds['temporal_delay'].generate_batch(val_batch_size)
        # X_t shape: [B, 10, 1], Y_t shape: [B, 10, 1]
        pred_t = self.model(X_t, 'temporal_delay')  # shape: [B, 10, 1]
        loss_t = self.criterion_temporal(pred_t, Y_t).item()
        
        # Convert logits to binary predictions
        pred_t_binary = (torch.sigmoid(pred_t) > 0.5).float()  # shape: [B, 10, 1]
        
        # Step-level accuracy (overall matches)
        step_correct = (pred_t_binary == Y_t)  # shape: [B, 10, 1]
        step_accuracy_t = step_correct.float().mean().item()
        
        # Sequence-level accuracy (entire sequence must match target)
        seq_correct = step_correct.all(dim=1).squeeze(-1)  # shape: [B]
        seq_accuracy_t = seq_correct.float().mean().item()
        
        metrics['temporal_delay'] = {
            'loss': loss_t, 
            'accuracy': step_accuracy_t, 
            'seq_accuracy': seq_accuracy_t
        }
        
        # --- Evaluate Task 3: ContextInversionWorld ---
        X_c, Y_c = self.worlds['context_inversion'].generate_batch(val_batch_size)
        # X_c shape: [B, 3], Y_c shape: [B, 1]
        pred_c = self.model(X_c, 'context_inversion')  # shape: [B, 1]
        loss_c = self.criterion_context(pred_c, Y_c).item()
        
        # Convert logits to binary predictions
        pred_c_binary = (torch.sigmoid(pred_c) > 0.5).float()  # shape: [B, 1]
        correct_c = (pred_c_binary == Y_c)  # shape: [B, 1]
        accuracy_c = correct_c.float().mean().item()
        
        metrics['context_inversion'] = {'loss': loss_c, 'accuracy': accuracy_c}
        
        # Record to history for plotting later
        self.history['spatial_shift']['loss'].append(loss_s)
        self.history['spatial_shift']['accuracy'].append(accuracy_s)
        
        self.history['temporal_delay']['loss'].append(loss_t)
        self.history['temporal_delay']['accuracy'].append(step_accuracy_t)
        self.history['temporal_delay']['seq_accuracy'].append(seq_accuracy_t)
        
        self.history['context_inversion']['loss'].append(loss_c)
        self.history['context_inversion']['accuracy'].append(accuracy_c)
        
        return metrics


class ProceduralIQTrainer:
    """
    Trainer designed to train an IQMicroUnit across 24 DSL operations
    using CrossEntropyLoss over the predicted program chain tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        world: AbstractWorld,
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.world = world
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        # History tracking for learning curves
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train_epoch(self, batch_size: int, steps_per_epoch: int, chain_depth: int = 2) -> float:
        """
        Runs one epoch of training using procedural batches, optimizing CrossEntropyLoss of program tokens.
        """
        self.model.train()
        device = next(self.model.parameters()).device
        total_loss = 0.0
        
        for _ in range(steps_per_epoch):
            self.optimizer.zero_grad()
            X_ctx, Y_ctx, X_qry, Y_qry, chain_ops = self.world.generate_batch(batch_size, chain_depth=chain_depth)
            
            # Send to model device
            X_ctx, Y_ctx = X_ctx.to(device), Y_ctx.to(device)
            X_qry, Y_qry = X_qry.to(device), Y_qry.to(device)
            chain_ops = chain_ops.to(device) # Shape: [B, 5]
            
            # Shift chain_ops by prepending START token (25) for teacher forcing
            start_tokens = torch.full((batch_size, 1), 25, dtype=torch.long, device=device)
            program_tokens = torch.cat([start_tokens, chain_ops[:, :-1]], dim=1) # Shape: [B, 5]
            
            pred = self.model(X_ctx, Y_ctx, X_qry, program_tokens=program_tokens) # Shape: [B, 5, 26]
            
            # pred.transpose(1, 2) has shape [B, 26, 5] for CrossEntropyLoss
            loss = self.criterion(pred.transpose(1, 2), chain_ops)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        mean_loss = total_loss / steps_per_epoch
        return mean_loss

    @torch.no_grad()
    def evaluate(self, val_batch_size: int = 512, chain_depth: int = 2) -> Tuple[float, float]:
        """
        Evaluates the model on unseen procedural rules.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        X_ctx, Y_ctx, X_qry, Y_qry, chain_ops = self.world.generate_batch(val_batch_size, chain_depth=chain_depth)
        
        X_ctx, Y_ctx = X_ctx.to(device), Y_ctx.to(device)
        X_qry, Y_qry = X_qry.to(device), Y_qry.to(device)
        chain_ops = chain_ops.to(device) # Shape: [B, 5]
        
        # 1. Compute validation loss using teacher forcing
        start_tokens = torch.full((val_batch_size, 1), 25, dtype=torch.long, device=device)
        teacher_forcing_inputs = torch.cat([start_tokens, chain_ops[:, :-1]], dim=1)
        pred_teacher = self.model(X_ctx, Y_ctx, X_qry, program_tokens=teacher_forcing_inputs) # [B, 5, 26]
        loss = self.criterion(pred_teacher.transpose(1, 2), chain_ops).item()
        
        # 2. Autoregressive prediction loop for evaluation metrics
        pred_tokens_list = []
        current_program = torch.full((val_batch_size, 1), 25, dtype=torch.long, device=device)
        
        for _ in range(5):
            logits = self.model(X_ctx, Y_ctx, X_qry, program_tokens=current_program) # [B, S, 26]
            next_token_logits = logits[:, -1, :] # Logits of the newly generated token -> [B, 26]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # [B, 1]
            pred_tokens_list.append(next_token)
            current_program = torch.cat([current_program, next_token], dim=1)
            
        pred_tokens = torch.cat(pred_tokens_list, dim=1) # [B, 5]
        
        # Strict Functional Sequence Accuracy (All-or-Nothing) using ExecutionEngine
        from execution_engine import ExecutionEngine
        y_pred = ExecutionEngine.execute_chain(X_qry, pred_tokens)
        
        # Check if predicted float output matches ground truth Y_qry within epsilon = 0.01
        correct_samples = torch.all(torch.abs(y_pred - Y_qry) < 0.01, dim=-1)
        seq_acc = correct_samples.float().mean().item()
        return loss, seq_acc


class IQMetaTrainer:
    """Stub class for backward compatibility after refactoring."""
    def __init__(self, *args, **kwargs):
        pass


