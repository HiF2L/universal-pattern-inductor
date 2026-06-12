# pyrefly: ignore [missing-import]
import torch
import random
from abc import ABC, abstractmethod
from typing import Tuple, List

class AbstractWorld(ABC):
    """Base class defining the interface for all micro-world environments."""

    @abstractmethod
    def generate_batch(self, batch_size: int, *args, **kwargs) -> Tuple:
        """
        Generates a batch of inputs and targets.

        Args:
            batch_size (int): The number of samples to generate in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of input and target tensors.
        """
        pass


class SpatialShiftWorld(AbstractWorld):
    """
    World 1: SpatialShiftWorld (Object Permanence)
    
    Generates a 5x5 binary grid with a 2x2 block of 1s placed randomly.
    The block shifts by 1 pixel (Left, Right, Up, or Down) and stays fully within the boundary.
    The input is the shifted grid, and the target is the center (x, y) of the shifted block.
    
    - Input Shape: [B, 1, 5, 5]
    - Target Shape: [B, 2]  (x_center, y_center)
    """

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X_list: List[torch.Tensor] = []
        Y_list: List[torch.Tensor] = []

        for _ in range(batch_size):
            # 1. Randomly sample initial top-left corner (r, c) in {0, 1, 2, 3}^2
            r = random.randint(0, 3)
            c = random.randint(0, 3)

            # 2. Identify all valid shift directions that keep the block fully within boundaries
            # Directions represented as (dr, dc)
            valid_shifts = []
            if r > 0:
                valid_shifts.append((-1, 0)) # Up
            if r < 3:
                valid_shifts.append((1, 0))  # Down
            if c > 0:
                valid_shifts.append((0, -1)) # Left
            if c < 3:
                valid_shifts.append((0, 1))  # Right

            # 3. Randomly select a valid shift direction
            dr, dc = random.choice(valid_shifts)
            r_new = r + dr
            c_new = c + dc

            # 4. Construct the 5x5 grid (post-shift state)
            grid = torch.zeros((1, 5, 5), dtype=torch.float32)
            grid[0, r_new : r_new + 2, c_new : c_new + 2] = 1.0

            # 5. Target center coordinates (x corresponds to column, y to row)
            x_center = c_new + 0.5
            y_center = r_new + 0.5
            target = torch.tensor([x_center, y_center], dtype=torch.float32)

            X_list.append(grid)
            Y_list.append(target)

        # Stack list into batch tensors
        X = torch.stack(X_list, dim=0)  # Shape: [B, 1, 5, 5]
        Y = torch.stack(Y_list, dim=0)  # Shape: [B, 2]

        return X, Y


class TemporalDelayWorld(AbstractWorld):
    """
    World 2: TemporalDelayWorld (Working Memory)
    
    Generates a sequence of length T=10 with 1 feature.
    A single 1 appears at a random time step t.
    The target is a sequence of length T=10 where a 1 appears exactly 3 steps later (at t+3).
    We restrict t to {0, ..., T-4} to ensure the target 1 is always within the sequence.
    
    - Input Shape: [B, T, 1]
    - Target Shape: [B, T, 1]
    """

    def __init__(self, sequence_length: int = 10, delay: int = 3):
        self.T = sequence_length
        self.delay = delay

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.zeros((batch_size, self.T, 1), dtype=torch.float32)
        Y = torch.zeros((batch_size, self.T, 1), dtype=torch.float32)

        # Restrict t so t + delay < T
        max_t = self.T - self.delay - 1

        for b in range(batch_size):
            t = random.randint(0, max_t)
            X[b, t, 0] = 1.0
            Y[b, t + self.delay, 0] = 1.0

        return X, Y


class ContextInversionWorld(AbstractWorld):
    """
    World 3: ContextInversionWorld (Rule Switching)
    
    Generates a 3-bit input vector.
    - Bit 0 and Bit 1 are data bits.
    - Bit 2 is the Context/Rule Bit.
    
    If Bit 2 == 0: Output = Bit 0 OR Bit 1
    If Bit 2 == 1: Output = Bit 0 XOR Bit 1
    
    - Input Shape: [B, 3]
    - Target Shape: [B, 1]
    """

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random bits (0 or 1) as float tensor
        bits = torch.randint(0, 2, (batch_size, 3), dtype=torch.float32)

        bit0 = bits[:, 0]
        bit1 = bits[:, 1]
        bit2 = bits[:, 2]

        # Rule calculations using element-wise logic
        or_result = (bit0.bool() | bit1.bool()).float()
        xor_result = (bit0.bool() ^ bit1.bool()).float()

        # Route output based on context (bit2)
        target = torch.where(bit2.bool(), xor_result, or_result).unsqueeze(1)  # Shape: [B, 1]

        return bits, target


class IQPatternWorld(AbstractWorld):
    """
    World for Phase 3: Crystallized Meta-IQ Benchmark.
    Generates batches containing context examples showing a transformation rule
    and a query object to apply that rule to.
    
    Rules:
    - 'arithmetic': scalar shift Delta in {-3, -2, -1, 1, 2, 3} applied element-wise.
    - 'cyclic': cyclic right shift k in {1, 2, 3, 4} positions.
    - 'bitwise': conditional bitwise XOR inversion based on a random mask.
    - 'inversion' [NEW]: sequence completely reversed in order (flip topology).
    - 'fibonacci' [NEW]: Fibonacci-like sequence where Y[i] = X[i] + X[i-1] (with X[-1]=0).
    - 'geometric' [NEW]: geometric scaling by alpha in {2, 3, -1}.
    - 'extremum' [NEW]: relational maximum filtering where Y[i] = max(X) for all i.
    - 'modulo' [NEW]: clock logic circular shift by constant Delta modulo 5.
    
    Returns:
        X_context: shape [B, 2, 5]
        Y_context: shape [B, 2, 5]
        X_query:   shape [B, 5]
        Y_query:   shape [B, 5]
    """

    def __init__(self, rule_type: str = 'arithmetic'):
        valid_rules = [
            'arithmetic', 'cyclic', 'bitwise', 'inversion', 'fibonacci', 
            'geometric', 'extremum', 'modulo', 'polarity', 'smoothing', 
            'sorting', 'delta', 'parity', 'cumsum', 'reflection'
        ]
        assert rule_type in valid_rules, f"Unknown rule: {rule_type}"
        self.rule_type = rule_type

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_context_list: List[torch.Tensor] = []
        Y_context_list: List[torch.Tensor] = []
        X_query_list: List[torch.Tensor] = []
        Y_query_list: List[torch.Tensor] = []

        for _ in range(batch_size):
            if self.rule_type == 'arithmetic':
                # Delta shift in {-3, -2, -1, 1, 2, 3}
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                
                # Sample continuous vectors
                x1 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0  # Shape: [5]
                x2 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0  # Shape: [5]
                x3 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0  # Shape: [5]

                y1 = x1 + delta  # Shape: [5]
                y2 = x2 + delta  # Shape: [5]
                y3 = x3 + delta  # Shape: [5]

            elif self.rule_type == 'cyclic':
                # Cyclic shift k in {1, 2, 3, 4}
                k = random.choice([1, 2, 3, 4])
                
                # Sample binary vectors
                x1 = torch.randint(0, 2, (5,), dtype=torch.float32)  # Shape: [5]
                x2 = torch.randint(0, 2, (5,), dtype=torch.float32)  # Shape: [5]
                x3 = torch.randint(0, 2, (5,), dtype=torch.float32)  # Shape: [5]

                y1 = torch.roll(x1, shifts=k, dims=0)  # Shape: [5]
                y2 = torch.roll(x2, shifts=k, dims=0)  # Shape: [5]
                y3 = torch.roll(x3, shifts=k, dims=0)  # Shape: [5]

            elif self.rule_type == 'bitwise':
                # Sample bitmask indicating indices to invert, must contain at least one 1
                mask = torch.randint(0, 2, (5,), dtype=torch.float32)
                while mask.sum() == 0:
                    mask = torch.randint(0, 2, (5,), dtype=torch.float32)
                
                # Sample binary vectors
                x1 = torch.randint(0, 2, (5,), dtype=torch.float32)  # Shape: [5]
                x2 = torch.randint(0, 2, (5,), dtype=torch.float32)  # Shape: [5]
                x3 = torch.randint(0, 2, (5,), dtype=torch.float32)  # Shape: [5]

                # Perform bitwise XOR via boolean casts
                y1 = (x1.bool() ^ mask.bool()).float()  # Shape: [5]
                y2 = (x2.bool() ^ mask.bool()).float()  # Shape: [5]
                y3 = (x3.bool() ^ mask.bool()).float()  # Shape: [5]

            elif self.rule_type == 'inversion':
                # Sequence Reversal
                # Sample binary vectors for BCE topology inversion
                x1 = torch.randint(0, 2, (5,), dtype=torch.float32)  # Shape: [5]
                x2 = torch.randint(0, 2, (5,), dtype=torch.float32)  # Shape: [5]
                x3 = torch.randint(0, 2, (5,), dtype=torch.float32)  # Shape: [5]

                y1 = torch.flip(x1, dims=[0])  # Shape: [5]
                y2 = torch.flip(x2, dims=[0])  # Shape: [5]
                y3 = torch.flip(x3, dims=[0])  # Shape: [5]

            elif self.rule_type == 'fibonacci':
                # Fibonacci-like addition logic (Y[i] = X[i] + X[i-1], X[-1]=0)
                # Sample continuous vectors in a small range to prevent exponential growth
                x1 = torch.rand(5, dtype=torch.float32) * 4.0 - 2.0  # Shape: [5]
                x2 = torch.rand(5, dtype=torch.float32) * 4.0 - 2.0  # Shape: [5]
                x3 = torch.rand(5, dtype=torch.float32) * 4.0 - 2.0  # Shape: [5]

                y1 = torch.zeros(5, dtype=torch.float32)
                y2 = torch.zeros(5, dtype=torch.float32)
                y3 = torch.zeros(5, dtype=torch.float32)

                y1[0] = x1[0]
                y2[0] = x2[0]
                y3[0] = x3[0]
                for i in range(1, 5):
                    y1[i] = x1[i] + x1[i - 1]
                    y2[i] = x2[i] + x2[i - 1]
                    y3[i] = x3[i] + x3[i - 1]

            elif self.rule_type == 'geometric':
                # Geometric scaling: multiplier alpha in {2, 3, -1}
                alpha = random.choice([2, 3, -1])

                x1 = torch.rand(5, dtype=torch.float32) * 4.0 - 2.0  # Shape: [5]
                x2 = torch.rand(5, dtype=torch.float32) * 4.0 - 2.0  # Shape: [5]
                x3 = torch.rand(5, dtype=torch.float32) * 4.0 - 2.0  # Shape: [5]

                y1 = x1 * alpha  # Shape: [5]
                y2 = x2 * alpha  # Shape: [5]
                y3 = x3 * alpha  # Shape: [5]

            elif self.rule_type == 'extremum':
                # Maximum Relational Filter: Y[i] = max(X) for all i
                x1 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0  # Shape: [5]
                x2 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0  # Shape: [5]
                x3 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0  # Shape: [5]

                y1 = torch.full((5,), torch.max(x1).item(), dtype=torch.float32)  # Shape: [5]
                y2 = torch.full((5,), torch.max(x2).item(), dtype=torch.float32)  # Shape: [5]
                y3 = torch.full((5,), torch.max(x3).item(), dtype=torch.float32)  # Shape: [5]

            elif self.rule_type == 'modulo':
                # Circular Clock arithmetic (Modulo 5 shift): (X + delta) % 5
                # Delta shift in {1, 2, 3, 4}
                delta = random.choice([1, 2, 3, 4])

                # Sample integer vectors in {0, 1, 2, 3, 4}
                x1_int = torch.randint(0, 5, (5,), dtype=torch.int32)
                x2_int = torch.randint(0, 5, (5,), dtype=torch.int32)
                x3_int = torch.randint(0, 5, (5,), dtype=torch.int32)

                # Target calculation modulo 5
                y1_int = (x1_int + delta) % 5
                y2_int = (x2_int + delta) % 5
                y3_int = (x3_int + delta) % 5

                # Scale inputs and targets by dividing by 4.0 so they are in [0, 1]
                # This mathematically enables proper training under nn.BCEWithLogitsLoss
                x1 = x1_int.float() / 4.0  # Shape: [5]
                x2 = x2_int.float() / 4.0  # Shape: [5]
                x3 = x3_int.float() / 4.0  # Shape: [5]

                y1 = y1_int.float() / 4.0  # Shape: [5]
                y2 = y2_int.float() / 4.0  # Shape: [5]
                y3 = y3_int.float() / 4.0  # Shape: [5]

            elif self.rule_type == 'polarity':
                # Even indices multiplied by -1.0, odd indices left unchanged
                x1 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x2 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x3 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0

                mask = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
                y1 = x1 * mask
                y2 = x2 * mask
                y3 = x3 * mask

            elif self.rule_type == 'smoothing':
                # Neighborhood average with circular boundary wrapping
                x1 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x2 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x3 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0

                y1 = torch.zeros(5, dtype=torch.float32)
                y2 = torch.zeros(5, dtype=torch.float32)
                y3 = torch.zeros(5, dtype=torch.float32)
                for i in range(5):
                    y1[i] = (x1[i] + x1[(i + 1) % 5]) / 2.0
                    y2[i] = (x2[i] + x2[(i + 1) % 5]) / 2.0
                    y3[i] = (x3[i] + x3[(i + 1) % 5]) / 2.0

            elif self.rule_type == 'sorting':
                # Sort elements in ascending order
                x1 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x2 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x3 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0

                y1, _ = torch.sort(x1)
                y2, _ = torch.sort(x2)
                y3, _ = torch.sort(x3)

            elif self.rule_type == 'delta':
                # Subtract minimum value of the vector from every element
                x1 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x2 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x3 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0

                y1 = x1 - torch.min(x1)
                y2 = x2 - torch.min(x2)
                y3 = x3 - torch.min(x3)

            elif self.rule_type == 'parity':
                # Count the number of 1s in binary vector. If odd, invert the vector.
                x1 = torch.randint(0, 2, (5,), dtype=torch.float32)
                x2 = torch.randint(0, 2, (5,), dtype=torch.float32)
                x3 = torch.randint(0, 2, (5,), dtype=torch.float32)

                y1 = (1.0 - x1) if (int(x1.sum().item()) % 2 == 1) else x1.clone()
                y2 = (1.0 - x2) if (int(x2.sum().item()) % 2 == 1) else x2.clone()
                y3 = (1.0 - x3) if (int(x3.sum().item()) % 2 == 1) else x3.clone()

            elif self.rule_type == 'cumsum':
                # Cumulative sum along elements, scaling down slightly to prevent huge values
                x1 = torch.rand(5, dtype=torch.float32) * 4.0 - 2.0
                x2 = torch.rand(5, dtype=torch.float32) * 4.0 - 2.0
                x3 = torch.rand(5, dtype=torch.float32) * 4.0 - 2.0

                y1 = torch.cumsum(x1, dim=0)
                y2 = torch.cumsum(x2, dim=0)
                y3 = torch.cumsum(x3, dim=0)

            elif self.rule_type == 'reflection':
                # Reflect vector elements around axis (equivalent to flip on continuous vectors)
                x1 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x2 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0
                x3 = torch.rand(5, dtype=torch.float32) * 10.0 - 5.0

                y1 = torch.flip(x1, dims=[0])
                y2 = torch.flip(x2, dims=[0])
                y3 = torch.flip(x3, dims=[0])

            # Collect Context Examples and Query Instances
            X_context_list.append(torch.stack([x1, x2], dim=0))  # Shape: [2, 5]
            Y_context_list.append(torch.stack([y1, y2], dim=0))  # Shape: [2, 5]
            X_query_list.append(x3)                             # Shape: [5]
            Y_query_list.append(y3)                             # Shape: [5]

        # Stack lists into unified batch tensors
        X_context = torch.stack(X_context_list, dim=0)  # Shape: [B, 2, 5]
        Y_context = torch.stack(Y_context_list, dim=0)  # Shape: [B, 2, 5]
        X_query = torch.stack(X_query_list, dim=0)      # Shape: [B, 5]
        Y_query = torch.stack(Y_query_list, dim=0)      # Shape: [B, 5]

        return X_context, Y_context, X_query, Y_query


class FractalIQEngine(AbstractWorld):
    """
    Procedural Rule Factory generating combinatorial micro-worlds for Phase 4.2.
    Chains randomized sequences of 24 operations on size-5 tensors.
    """
    
    def generate_batch(self, batch_size: int, chain_depth: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_ctx_list = []
        Y_ctx_list = []
        X_qry_list = []
        Y_qry_list = []
        chain_ops_list = []
        
        # Import dynamically to avoid circular import issues
        from execution_engine import ExecutionEngine
        
        for _ in range(batch_size):
            # 1. Procedurally generate sequence of operations (excluding PAD=0 during sampling)
            chain = [random.randint(1, 23) for _ in range(chain_depth)]
            
            # Append STOP = 24, and Pad the chain with 0 (PAD) to a fixed length of 5
            padded_chain = chain + [24] + [0] * (5 - len(chain) - 1)
            chain_ops_list.append(torch.tensor(padded_chain, dtype=torch.long))
            
            # 2. Decide continuous vs binary input style for generalization
            is_binary = (random.random() < 0.5)
            
            def make_vec():
                if is_binary:
                    return torch.randint(0, 2, (5,), dtype=torch.float32)
                else:
                    return torch.rand(5, dtype=torch.float32) * 8.0 - 4.0
            
            # Stack all 5 vectors to process them batch-wise using the execution engine
            x1 = make_vec()
            x2 = make_vec()
            x3 = make_vec()
            x4 = make_vec()
            x5 = make_vec()
            
            x_all = torch.stack([x1, x2, x3, x4, x5], dim=0) # Shape: [5, 5]
            
            # 3. Apply operational chain batch-wise for speed
            y_all = ExecutionEngine.execute_chain(x_all, chain)
            
            y1, y2, y3, y4, y5 = y_all[0], y_all[1], y_all[2], y_all[3], y_all[4]
            
            X_ctx_list.append(torch.stack([x1, x2, x3, x4], dim=0))  # Shape: [4, 5]
            Y_ctx_list.append(torch.stack([y1, y2, y3, y4], dim=0))  # Shape: [4, 5]
            X_qry_list.append(x5)
            Y_qry_list.append(y5)
            
        X_context = torch.stack(X_ctx_list, dim=0)  # Shape: [B, 4, 5]
        Y_context = torch.stack(Y_ctx_list, dim=0)  # Shape: [B, 4, 5]
        X_query = torch.stack(X_qry_list, dim=0)      # Shape: [B, 5]
        Y_query = torch.stack(Y_qry_list, dim=0)      # Shape: [B, 5]
        chain_ops = torch.stack(chain_ops_list, dim=0)  # Shape: [B, 3]
        
        return X_context, Y_context, X_query, Y_query, chain_ops

