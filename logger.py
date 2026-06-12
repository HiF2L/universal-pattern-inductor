# pyrefly: ignore [missing-import]
import torch
import time
import random
from execution_engine import ExecutionEngine

OP_NAMES = {
    0: "PAD (no-op)",
    1: "REVERSE (flip array)",
    2: "SHIFT_R (cyclic shift right)",
    3: "SHIFT_L (cyclic shift left)",
    4: "SWAP_HALVES (swap first 2 and last 3 elements)",
    5: "SORT_ASC (sort ascending)",
    6: "SORT_DESC (sort descending)",
    7: "CUMSUM (cumulative sum)",
    8: "CUMPROD (cumulative product)",
    9: "DIFF (adjacent difference)",
    10: "RUNNING_MEAN (running average)",
    11: "INVERT_SIGN (multiply by -1)",
    12: "ABS (absolute value)",
    13: "ROUND (round to nearest integer)",
    14: "CEIL (ceiling)",
    15: "FLOOR (floor)",
    16: "LOG_TRANSFORM (sign(x) * log(|x| + 1))",
    17: "MASK_GT_ZERO (keep if > 0 else 0)",
    18: "MASK_LT_ZERO (keep if < 0 else 0)",
    19: "BINARIZE (1 if > 0 else 0)",
    20: "INVERT_MASK (1 if == 0 else 0)",
    21: "CLAMP_UNIT (clamp to [-1, 1])",
    22: "ARGMAX_ONEHOT (one-hot of max element)",
    23: "ARGMIN_ONEHOT (one-hot of min element)",
    24: "STOP",
    25: "UNUSED"
}

class CoTLogger:
    def __init__(self, filepath: str = r"c:\Users\Hitori US\Desktop\Dev\Intellegence bit\cot_brain_debug.log"):
        self.filepath = filepath
        # Clear or initialize log file
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("=== Discrete Chain-of-Thought Reasoning Logger Initialized ===\n\n")

    def format_vector(self, t: torch.Tensor) -> str:
        vals = []
        for val in t.tolist():
            if abs(val - round(val)) < 1e-4:
                vals.append(str(int(round(val))))
            else:
                vals.append(f"{val:.2f}")
        return "[" + ", ".join(vals) + "]"

    def log_validation_episodes(self, epoch: int, model: torch.nn.Module, X_ctx, Y_ctx, X_qry, Y_qry, chain_ops):
        """
        X_ctx: [B, 4, 5]
        Y_ctx: [B, 4, 5]
        X_qry: [B, 5]
        Y_qry: [B, 5]
        chain_ops: [B, 5]
        """
        model.eval()
        device = next(model.parameters()).device
        B = X_ctx.size(0)
        
        # Select 5 random indices
        indices = random.sample(range(B), min(5, B))
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"\n============================================================\n")
            f.write(f" VALIDATION DEEP BREAKDOWN - EPOCH {epoch:04d} | {timestamp}\n")
            f.write(f"============================================================\n")
            
            # Predict step-by-step autoregressively
            with torch.no_grad():
                X_ctx_dev = X_ctx.to(device)
                Y_ctx_dev = Y_ctx.to(device)
                X_qry_dev = X_qry.to(device)
                
                current_program = torch.full((B, 1), 25, dtype=torch.long, device=device)
                all_step_probs = []
                pred_tokens_list = []
                
                for step in range(5):
                    logits = model(X_ctx_dev, Y_ctx_dev, X_qry_dev, program_tokens=current_program) # [B, S, 26]
                    next_token_logits = logits[:, -1, :] # [B, 26]
                    step_probs = torch.softmax(next_token_logits, dim=-1) # [B, 26]
                    all_step_probs.append(step_probs)
                    
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # [B, 1]
                    pred_tokens_list.append(next_token)
                    current_program = torch.cat([current_program, next_token], dim=1)
                
                probs = torch.stack(all_step_probs, dim=1) # [B, 5, 26]
                pred_tokens = torch.cat(pred_tokens_list, dim=1) # [B, 5]
                
            for idx in indices:
                ep_id = f"Ep-{epoch:04d}-{idx:03d}-{random.randint(1000, 9999)}"
                f.write(f"\n[Episode ID: {ep_id}]\n")
                f.write(f"------------------------------------------------------------\n")
                
                # 1. Ground Truth Chain Ops
                gt_chain = [op for op in chain_ops[idx].tolist() if op != 0]
                gt_chain_names = [OP_NAMES.get(op, f"OP_{op}") for op in gt_chain]
                f.write(f"  Ground Truth Chain: {' -> '.join(gt_chain_names)}\n\n")
                
                # 2. Context Pairs
                f.write(f"  Context Examples:\n")
                for c in range(4):
                    x_str = self.format_vector(X_ctx[idx, c])
                    y_str = self.format_vector(Y_ctx[idx, c])
                    f.write(f"    Example {c+1}: {x_str} ===> {y_str}\n")
                f.write("\n")
                
                # 3. Step-by-Step Token Prediction with Top-3 analysis
                f.write(f"  Step-by-Step Prediction Analysis:\n")
                for step in range(5):
                    step_probs = probs[idx, step] # [26]
                    top_probs, top_indices = torch.topk(step_probs, k=3)
                    
                    top_3_str = []
                    for val, i_tok in zip(top_probs.tolist(), top_indices.tolist()):
                        name = OP_NAMES.get(i_tok, f"OP_{i_tok}")
                        top_3_str.append(f"{name}: {val*100:.2f}%")
                        
                    selected_token = pred_tokens[idx, step].item()
                    selected_name = OP_NAMES.get(selected_token, f"OP_{selected_token}")
                    
                    f.write(f"    Step {step+1}: Predicted: {selected_name} | Top-3 Probabilities: {', '.join(top_3_str)}\n")
                f.write("\n")
                
                # 4. Final Output from Python execution engine and target matching
                tokens_list = pred_tokens[idx].tolist()
                y_pred = ExecutionEngine.execute_chain(X_qry[idx].to(device), tokens_list)
                
                x_qry_str = self.format_vector(X_qry[idx])
                y_pred_str = self.format_vector(y_pred)
                y_true_str = self.format_vector(Y_qry[idx])
                
                is_correct = torch.all(torch.abs(y_pred - Y_qry[idx].to(device)) < 0.01).item()
                
                f.write(f"  Query Input:  {x_qry_str}\n")
                f.write(f"  Predicted Y:  {y_pred_str}\n")
                f.write(f"  True Target:  {y_true_str}\n")
                f.write(f"  Match Status: {'SUCCESS' if is_correct else 'FAILURE'}\n")
                f.write(f"============================================================\n")

    def log_exam_summary(self, accuracy: float, avg_loss: float):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"\n============================================================\n")
            f.write(f" FINAL ZERO-SHOT OOD EXAM SUMMARY | {timestamp}\n")
            f.write(f"============================================================\n")
            f.write(f"  Zero-Shot Holdout Accuracy (Chain Depth 3): {accuracy*100:.2f}%\n")
            f.write(f"  Zero-Shot Average DSL CrossEntropy Loss  : {avg_loss:.4f}\n")
            f.write(f"============================================================\n")
