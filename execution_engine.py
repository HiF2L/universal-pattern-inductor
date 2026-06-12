# pyrefly: ignore [missing-import]
import torch

class ExecutionEngine:
    """
    DSL Execution Engine for Phase 4.2.
    Implements 24 operations that map 5-element tensors (batched or single)
    to target representations.
    """

    @staticmethod
    def execute_op(x: torch.Tensor, op_id: int) -> torch.Tensor:
        """
        Executes a single operation on tensor x (shape: [..., 5])
        """
        out = x.clone().float()
        
        if op_id == 0 or op_id >= 24:  # PAD or STOP / unused
            pass
        elif op_id == 1:  # REVERSE
            out = torch.flip(out, dims=[-1])
        elif op_id == 2:  # SHIFT_R
            out = torch.roll(out, shifts=1, dims=-1)
        elif op_id == 3:  # SHIFT_L
            out = torch.roll(out, shifts=-1, dims=-1)
        elif op_id == 4:  # SWAP_HALVES
            # Swap first 2 and last 3 elements (seq len is 5)
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
            # diff of adjacent elements, first element is preserved
            first = out[..., :1]
            rest = out[..., 1:] - out[..., :-1]
            out = torch.cat([first, rest], dim=-1)
        elif op_id == 10:  # RUNNING_MEAN
            # running mean of two adjacent elements, Y[0] = X[0]
            first = out[..., :1]
            rest = (out[..., 1:] + out[..., :-1]) / 2.0
            out = torch.cat([first, rest], dim=-1)
        elif op_id == 11:  # INVERT_SIGN
            out = -out
        elif op_id == 12:  # ABS
            out = torch.abs(out)
        elif op_id == 13:  # ROUND
            out = torch.round(out)
        elif op_id == 14:  # CEIL
            out = torch.ceil(out)
        elif op_id == 15:  # FLOOR
            out = torch.floor(out)
        elif op_id == 16:  # LOG_TRANSFORM
            out = torch.sign(out) * torch.log(torch.abs(out) + 1.0)
        elif op_id == 17:  # MASK_GT_ZERO
            out = torch.where(out > 0.0, out, torch.zeros_like(out))
        elif op_id == 18:  # MASK_LT_ZERO
            out = torch.where(out < 0.0, out, torch.zeros_like(out))
        elif op_id == 19:  # BINARIZE
            out = torch.where(out > 0.0, torch.ones_like(out), torch.zeros_like(out))
        elif op_id == 20:  # INVERT_MASK
            out = torch.where(out == 0.0, torch.ones_like(out), torch.zeros_like(out))
        elif op_id == 21:  # CLAMP_UNIT
            out = torch.clamp(out, -1.0, 1.0)
        elif op_id == 22:  # ARGMAX_ONEHOT
            idx = torch.argmax(out, dim=-1, keepdim=True)
            onehot = torch.zeros_like(out)
            onehot.scatter_(-1, idx, 1.0)
            out = onehot
        elif op_id == 23:  # ARGMIN_ONEHOT
            idx = torch.argmin(out, dim=-1, keepdim=True)
            onehot = torch.zeros_like(out)
            onehot.scatter_(-1, idx, 1.0)
            out = onehot
            
        return out

    @classmethod
    def execute_chain(cls, x: torch.Tensor, chain) -> torch.Tensor:
        """
        Executes a chain of operations on tensor x.
        Supports:
          - A list/tuple/1D tensor of operation IDs applied to all elements of x.
          - A 2D tensor of shape [B, T] containing different operation chains for each batch element.
        """
        if isinstance(chain, (list, tuple)):
            out = x.clone().float()
            for op_id in chain:
                if isinstance(op_id, torch.Tensor):
                    op_id = int(op_id.item())
                out = cls.execute_op(out, op_id)
            return out

        if isinstance(chain, torch.Tensor):
            if chain.dim() == 1:
                out = x.clone().float()
                for op_id in chain:
                    out = cls.execute_op(out, int(op_id.item()))
                return out
            elif chain.dim() == 2:
                # Shape check/assertion: x should match batch size of chain
                out = x.clone().float()
                B, T = chain.shape
                # Track which samples are still active (i.e. have not hit STOP or PAD yet)
                active = torch.ones(B, dtype=torch.bool, device=chain.device)
                for t in range(T):
                    op_ids = chain[:, t]  # Shape: [B]
                    next_out = out.clone()
                    # Vectorized batch execution: apply each operation type to active sub-batches
                    for op_id in range(24):
                        mask = (op_ids == op_id) & active
                        if not mask.any():
                            continue
                        next_out[mask] = cls.execute_op(out[mask], op_id)
                    # Update active mask: turn off if op_id is STOP (24) or PAD (0)
                    active = active & (op_ids != 24) & (op_ids != 0)
                    out = next_out
                return out

        raise ValueError(f"Unsupported chain type or shape: {type(chain)}")
