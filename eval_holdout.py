# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
import random
# pyrefly: ignore [missing-import]
import numpy as np
import os
from typing import Tuple

from model import IQMicroUnit

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class MirrorCyclicWorld:
    """
    Unseen World A: Composition of Reflection and Cyclic Shift.
    
    Hidden Rule: Reverse the binary vector completely, then cyclically shift to the right by k positions.
    - Input: Binary vector of size 5
    - Target: Binary vector of size 5
    """
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_context_list = []
        Y_context_list = []
        X_query_list = []
        Y_query_list = []
        
        for _ in range(batch_size):
            # Constant cyclic shift k per episode
            k = random.choice([1, 2, 3, 4])
            
            # Binary vectors
            x1 = torch.randint(0, 2, (5,), dtype=torch.float32)
            x2 = torch.randint(0, 2, (5,), dtype=torch.float32)
            x3 = torch.randint(0, 2, (5,), dtype=torch.float32)
            
            # Y = cyclic_shift(flip(X))
            y1 = torch.roll(torch.flip(x1, dims=[0]), shifts=k, dims=0)
            y2 = torch.roll(torch.flip(x2, dims=[0]), shifts=k, dims=0)
            y3 = torch.roll(torch.flip(x3, dims=[0]), shifts=k, dims=0)
            
            X_context_list.append(torch.stack([x1, x2], dim=0))
            Y_context_list.append(torch.stack([y1, y2], dim=0))
            X_query_list.append(x3)
            Y_query_list.append(y3)
            
        X_context = torch.stack(X_context_list, dim=0)
        Y_context = torch.stack(Y_context_list, dim=0)
        X_query = torch.stack(X_query_list, dim=0)
        Y_query = torch.stack(Y_query_list, dim=0)
        
        return X_context, Y_context, X_query, Y_query

class InterleavedPolarityProgressionWorld:
    """
    Unseen World B: Composition of Polarity logic and Arithmetic Progression.
    
    Hidden Rule: Shift vector elements by a constant Delta, and multiply elements at even indices by -1.
    - Input: Continuous vector of size 5
    - Target: Continuous vector of size 5
    """
    def __init__(self):
        self.mask = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32)

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_context_list = []
        Y_context_list = []
        X_query_list = []
        Y_query_list = []
        
        for _ in range(batch_size):
            # Constant delta progression per episode
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            
            # Continuous vectors
            x1 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
            x2 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
            x3 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
            
            # Y = (X + delta) * mask
            y1 = (x1 + delta) * self.mask
            y2 = (x2 + delta) * self.mask
            y3 = (x3 + delta) * self.mask
            
            X_context_list.append(torch.stack([x1, x2], dim=0))
            Y_context_list.append(torch.stack([y1, y2], dim=0))
            X_query_list.append(x3)
            Y_query_list.append(y3)
            
        X_context = torch.stack(X_context_list, dim=0)
        Y_context = torch.stack(Y_context_list, dim=0)
        X_query = torch.stack(X_query_list, dim=0)
        Y_query = torch.stack(Y_query_list, dim=0)
        
        return X_context, Y_context, X_query, Y_query

def evaluate_holdout(model_path: str = "cot_best_model.pt", num_episodes: int = 1000):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading IQMicroUnit framework onto device: {device}")
    
    # Instantiate scaled capacity architecture (hidden_dim=256)
    model = IQMicroUnit(hidden_dim=256)
    
    # Load model weights from target path
    if not os.path.exists(model_path):
        print(f"[-] Error: Checkpoint file '{model_path}' not found in current directory.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("[+] Model loaded successfully. Initializing unseen test environments...")
    
    world_A = MirrorCyclicWorld()
    world_B = InterleavedPolarityProgressionWorld()
    
    X_ctx_A, Y_ctx_A, X_qry_A, Y_qry_A = world_A.generate_batch(num_episodes)
    X_ctx_B, Y_ctx_B, X_qry_B, Y_qry_B = world_B.generate_batch(num_episodes)
    
    # Send validation tensors to proper device (keep as floats)
    X_ctx_A, Y_ctx_A, X_qry_A, Y_qry_A = X_ctx_A.to(device), Y_ctx_A.to(device), X_qry_A.to(device), Y_qry_A.to(device)
    X_ctx_B, Y_ctx_B, X_qry_B, Y_qry_B = X_ctx_B.to(device), Y_ctx_B.to(device), X_qry_B.to(device), Y_qry_B.to(device)
    
    print(f"[+] Running zero-shot evaluation across {num_episodes} test episodes...\n")
    
    print("=" * 60)
    print("UNSEEN WORLD A: MirrorCyclicWorld (Composition of Reflection + Shift)")
    print("=" * 60)
    
    from execution_engine import ExecutionEngine
    
    with torch.no_grad():
        # Autoregressive prediction loop for World A
        pred_tokens_list = []
        current_program = torch.full((num_episodes, 1), 25, dtype=torch.long, device=device)
        for _ in range(5):
            logits = model(X_ctx_A, Y_ctx_A, X_qry_A, program_tokens=current_program)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            pred_tokens_list.append(next_token)
            current_program = torch.cat([current_program, next_token], dim=1)
        pred_tokens = torch.cat(pred_tokens_list, dim=1) # Shape: [B, 5]
        
        y_pred = ExecutionEngine.execute_chain(X_qry_A, pred_tokens)
        correct_samples = torch.all(torch.abs(y_pred - Y_qry_A) < 0.01, dim=-1)
        acc_seq = correct_samples.float().mean().item()
        print(f"-> Zero-Shot Functional Sequence Accuracy = {acc_seq*100:6.2f}%")
            
    print("\n" + "=" * 60)
    print("UNSEEN WORLD B: Interleaved PolarityProgression (Composition of Polarity + Arithmetic)")
    print("=" * 60)
    
    with torch.no_grad():
        # Autoregressive prediction loop for World B
        pred_tokens_list = []
        current_program = torch.full((num_episodes, 1), 25, dtype=torch.long, device=device)
        for _ in range(5):
            logits = model(X_ctx_B, Y_ctx_B, X_qry_B, program_tokens=current_program)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            pred_tokens_list.append(next_token)
            current_program = torch.cat([current_program, next_token], dim=1)
        pred_tokens = torch.cat(pred_tokens_list, dim=1) # Shape: [B, 5]
        
        y_pred = ExecutionEngine.execute_chain(X_qry_B, pred_tokens)
        correct_samples = torch.all(torch.abs(y_pred - Y_qry_B) < 0.01, dim=-1)
        acc_seq = correct_samples.float().mean().item()
        print(f"-> Zero-Shot Functional Sequence Accuracy = {acc_seq*100:6.2f}%")
            
    print("\nEvaluation pipeline complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Zero-shot holdout evaluation")
    parser.add_argument("--model_path", type=str, default="cot_best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of evaluation episodes")
    args = parser.parse_args()
    
    evaluate_holdout(model_path=args.model_path, num_episodes=args.num_episodes)
