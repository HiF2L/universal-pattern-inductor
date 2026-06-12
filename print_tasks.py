import torch
import random
from worlds import FractalIQEngine

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
    24: "STOP (end chain)"
}

def format_tensor(t):
    vals = []
    for val in t.tolist():
        if abs(val - round(val)) < 1e-4:
            vals.append(str(int(round(val))))
        else:
            vals.append(f"{val:.2f}")
    return "[" + ", ".join(vals) + "]"

def main():
    # Set seed for reproducibility of tasks shown
    random.seed(random.randint(0, 10000))
    torch.manual_seed(random.randint(0, 10000))
    
    engine = FractalIQEngine()
    
    print("=" * 80)
    print("         GENERATING 10 RANDOM PROCEDURAL META-LEARNING EPISODES (TASKS)       ")
    print("=" * 80)
    
    # Generate 5 tasks with depth 2 and 5 tasks with depth 3
    batches = [
        engine.generate_batch(batch_size=5, chain_depth=2),
        engine.generate_batch(batch_size=5, chain_depth=3)
    ]
    
    task_idx = 1
    for X_context, Y_context, X_query, Y_query, chain_ops in batches:
        B = X_context.size(0)
        for i in range(B):
            chain = chain_ops[i].tolist()
            # Filter out padding operations (0) for cleaner operation logging
            active_chain = [op for op in chain if op != 0]
            
            # Identify if the context contains binary inputs (all elements are 0 or 1)
            is_binary = torch.all((X_context[i] == 0.0) | (X_context[i] == 1.0)).item()
            
            print(f"\n[TASK #{task_idx}] Chain Depth: {len(active_chain)} | Mode: {'Binary' if is_binary else 'Continuous'}")
            print("-" * 60)
            print("  Hidden Operation Chain:")
            for op_idx, op_id in enumerate(active_chain):
                print(f"    Step {op_idx + 1}: {OP_NAMES.get(op_id, 'Unknown Op')}")
            
            print("\n  Context Examples (Model learns the rule from these):")
            for ctx_idx in range(4):
                x_str = format_tensor(X_context[i, ctx_idx])
                y_str = format_tensor(Y_context[i, ctx_idx])
                print(f"    Example {ctx_idx + 1}: {x_str}  ===>  {y_str}")
            
            print("\n  Query instance (Model must solve this):")
            print(f"    Input:  {format_tensor(X_query[i])}")
            print(f"    Target: {format_tensor(Y_query[i])}")
            print("=" * 80)
            
            task_idx += 1

if __name__ == '__main__':
    main()
