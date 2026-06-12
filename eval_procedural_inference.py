# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
import os
import random
import numpy as np
from typing import Tuple

from model import IQMicroUnit
from worlds import FractalIQEngine
from execution_engine import ExecutionEngine

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def differentiable_execute_op(x: torch.Tensor, op_id: int) -> torch.Tensor:
    """
    Executes a single operation with Straight-Through Estimators (STE) for non-differentiable ops.
    """
    out = x.clone().float()
    
    if op_id == 0:  # PAD
        pass
    elif op_id == 1:  # REVERSE
        out = torch.flip(out, dims=[-1])
    elif op_id == 2:  # SHIFT_R
        out = torch.roll(out, shifts=1, dims=-1)
    elif op_id == 3:  # SHIFT_L
        out = torch.roll(out, shifts=-1, dims=-1)
    elif op_id == 4:  # SWAP_HALVES
        out = torch.cat([out[..., 2:], out[..., :2]], dim=-1)
    elif op_id == 5:  # SORT_ASC
        out, _ = torch.sort(out, dim=-1)
    elif op_id == 6:  # SORT_DESC
        out, _ = torch.sort(out, dim=-1, descending=True)
    elif op_id == 7:  # CUMSUM
        out = torch.cumsum(out, dim=-1)
    elif op_id == 8:  # CUMPROD
        out = torch.cumprod(out, dim=-1)
    elif op_id == 9:  # DIFF
        first = out[..., :1]
        rest = out[..., 1:] - out[..., :-1]
        out = torch.cat([first, rest], dim=-1)
    elif op_id == 10:  # RUNNING_MEAN
        first = out[..., :1]
        rest = (out[..., 1:] + out[..., :-1]) / 2.0
        out = torch.cat([first, rest], dim=-1)
    elif op_id == 11:  # INVERT_SIGN
        out = -out
    elif op_id == 12:  # ABS
        out = torch.abs(out)
    elif op_id == 13:  # ROUND
        target = torch.round(out)
        out = out + (target - out).detach()
    elif op_id == 14:  # CEIL
        target = torch.ceil(out)
        out = out + (target - out).detach()
    elif op_id == 15:  # FLOOR
        target = torch.floor(out)
        out = out + (target - out).detach()
    elif op_id == 16:  # LOG_TRANSFORM
        out = torch.sign(out) * torch.log(torch.abs(out) + 1.0)
    elif op_id == 17:  # MASK_GT_ZERO
        out = torch.where(out > 0.0, out, torch.zeros_like(out))
    elif op_id == 18:  # MASK_LT_ZERO
        out = torch.where(out < 0.0, out, torch.zeros_like(out))
    elif op_id == 19:  # BINARIZE
        target = torch.where(out > 0.0, torch.ones_like(out), torch.zeros_like(out))
        out = out + (target - out).detach()
    elif op_id == 20:  # INVERT_MASK
        target = torch.where(out == 0.0, torch.ones_like(out), torch.zeros_like(out))
        out = out + (target - out).detach()
    elif op_id == 21:  # CLAMP_UNIT
        out = torch.clamp(out, -1.0, 1.0)
    elif op_id == 22:  # ARGMAX_ONEHOT
        idx = torch.argmax(out, dim=-1, keepdim=True)
        onehot = torch.zeros_like(out)
        onehot.scatter_(-1, idx, 1.0)
        out = out + (onehot - out).detach()
    elif op_id == 23:  # ARGMIN_ONEHOT
        idx = torch.argmin(out, dim=-1, keepdim=True)
        onehot = torch.zeros_like(out)
        onehot.scatter_(-1, idx, 1.0)
        out = out + (onehot - out).detach()
        
    return out

def soft_execute_chain(x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """
    Computes a soft/differentiable execution of the 24 operations on tensor x.
    Gradients can flow back from the output to the probability tensor.
    x: [B, 4, 5] or [B, 5]
    probs: [B, 3, 24]
    """
    out = x.clone().float()
    for t in range(3):
        # Compute all 24 operations on current state
        op_outputs = []
        for op_id in range(24):
            op_outputs.append(differentiable_execute_op(out, op_id))
        
        # Stack operations along the last dimension -> [..., 5, 24]
        stacked = torch.stack(op_outputs, dim=-1)
        
        # Extract probabilities for step t -> [B, 24]
        p = probs[:, t]
        
        # Broadcast probabilities across middle dimensions
        for _ in range(stacked.dim() - 2):
            p = p.unsqueeze(1)  # e.g., becomes [B, 1, 1, 24]
            
        # Weighted sum over operations
        out = (stacked * p).sum(dim=-1)
        
    return out

def differentiable_inference(
    model: nn.Module, 
    X_context: torch.Tensor, 
    Y_context: torch.Tensor, 
    X_query: torch.Tensor, 
    num_steps: int = 15, 
    lr: float = 0.05
) -> torch.Tensor:
    """
    Optimizes the latent rule representation using gradient descent on context inputs and targets.
    """
    # 1. Freeze Weights
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    # 2. Extract Initial Latent State (Query Token Representation)
    # Replicating model forward path
    ctx_pairs = torch.cat([X_context, Y_context], dim=-1) # [B, 4, 10]
    ctx_proj = model.universal_projector(ctx_pairs)       # [B, 4, 256]
    qry_proj = model.query_projector(X_query).unsqueeze(1) # [B, 1, 256]
    seq = torch.cat([ctx_proj, qry_proj], dim=1)           # [B, 5, 256]
    
    # Add positional embeddings
    T_seq = seq.size(1)
    pos = torch.cat([model.pos_embedding[:, :T_seq-1, :], model.pos_embedding[:, -1:, :]], dim=1)
    seq = seq + pos
    
    # Transformer Encoder Core
    out_seq = model.core_transformer(seq)
    latent_rule = out_seq[:, 4, :] # Extract the query token representation, shape: [B, 256]
    
    # 3. Convert to Learnable Parameter
    latent_param = nn.Parameter(latent_rule.clone().detach())
    
    # 4. Setup Inference Optimizer
    optimizer = torch.optim.Adam([latent_param], lr=lr)
    
    # 5. Optimization Loop (Reasoning Steps)
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Decode current latent representation and reshape to [B, 3, 24]
        logits_flat = model.universal_decoder(latent_param)
        logits = logits_flat.view(-1, 3, 24)
        probs = torch.softmax(logits, dim=-1)
        
        # Compute soft context predictions
        context_predictions = soft_execute_chain(X_context, probs)
        
        # Compute loss on context targets
        loss = nn.MSELoss()(context_predictions, Y_context)
        
        # Backward and step
        loss.backward()
        optimizer.step()
        
    # 6. Final Output
    with torch.no_grad():
        final_logits_flat = model.universal_decoder(latent_param)
        final_logits = final_logits_flat.view(-1, 3, 24)
        
    return final_logits

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")
    
    # Locate model checkpoint
    model_path = "procedural_best_model.pt"
    if not os.path.exists(model_path):
        model_path = "best_model.pt"
        
    if not os.path.exists(model_path):
        print(f"[-] Error: Checkpoint file '{model_path}' not found in current directory.")
        return
        
    print(f"[+] Loading model weights from: {model_path}")
    model = IQMicroUnit(hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Instantiate engine and generate OOD batch (chain depth 3)
    print("[+] Generating 512 OOD task episodes (Chain Depth 3)...")
    engine = FractalIQEngine()
    X_context, Y_context, X_query, Y_query, chain_ops = engine.generate_batch(batch_size=512, chain_depth=3)
    
    # Move tensors to device
    X_context = X_context.to(device)
    Y_context = Y_context.to(device)
    X_query = X_query.to(device)
    Y_query = Y_query.to(device)
    chain_ops = chain_ops.to(device)
    
    # Define criterion for loss tracking
    criterion = nn.CrossEntropyLoss()
    
    # --- Pass 1: Standard Inference ---
    print("[+] Running Pass 1: Standard Inference (Zero-Shot)...")
    with torch.no_grad():
        pred_y_std = model(X_context, Y_context, X_query) # [B, 3, 24]
        loss_std = criterion(pred_y_std.transpose(1, 2), chain_ops).item()
        
        pred_tokens_std = torch.argmax(pred_y_std, dim=-1) # [B, 3]
        y_pred_std = ExecutionEngine.execute_chain(X_query, pred_tokens_std)
        correct_std = torch.all(torch.abs(y_pred_std - Y_query) < 0.01, dim=-1)
        acc_std = correct_std.float().mean().item()
        
    # --- Pass 2: Differentiable Inference (TTO) ---
    print("[+] Running Pass 2: Differentiable Inference (TTO with 15 steps)...")
    pred_y_opt = differentiable_inference(model, X_context, Y_context, X_query, num_steps=15, lr=0.05)
    
    with torch.no_grad():
        loss_opt = criterion(pred_y_opt.transpose(1, 2), chain_ops).item()
        pred_tokens_opt = torch.argmax(pred_y_opt, dim=-1) # [B, 3]
        y_pred_opt = ExecutionEngine.execute_chain(X_query, pred_tokens_opt)
        correct_opt = torch.all(torch.abs(y_pred_opt - Y_query) < 0.01, dim=-1)
        acc_opt = correct_opt.float().mean().item()
        
    # Print results table
    print("\n" + "=" * 65)
    print("                TEST-TIME OPTIMIZATION COMPARISON")
    print("=" * 65)
    print(f"  Inference Method     |  Accuracy  |  DSL CrossEntropy Loss")
    print("-" * 65)
    print(f"  Standard Inference   |   {acc_std*100:6.2f}%   |         {loss_std:.4f}")
    print(f"  Differentiable Inf   |   {acc_opt*100:6.2f}%   |         {loss_opt:.4f}")
    print("=" * 65)
    print(f"  Improvement          |  {(acc_opt - acc_std)*100:+5.2f}%   |         {loss_opt - loss_std:+.4f}")
    print("=" * 65)
    
if __name__ == '__main__':
    main()
