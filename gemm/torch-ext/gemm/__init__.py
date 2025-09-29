from typing import Optional
import torch
from ._ops import ops

def gemm(a: torch.Tensor, b: torch.Tensor, as_: torch.Tensor, bs: torch.Tensor, 
         out: Optional[torch.Tensor] = None) -> torch.Tensor:
         
    if out is None:
        # Create output tensor with appropriate shape and dtype
        M, K = a.shape
        K_b, N = b.shape
        assert K == K_b, f"Matrix dimension mismatch: A has {K} cols, B has {K_b} rows"
        
        # Output should be BF16 type on the same device as inputs
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    
    ops.gemm(out, a, b, as_, bs)
    return out

