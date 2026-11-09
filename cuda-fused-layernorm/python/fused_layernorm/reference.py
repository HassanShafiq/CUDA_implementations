# SPDX-License-Identifier: MIT
"""Reference implementations used by the tests and the benchmark (no custom kernels)."""
from __future__ import annotations

from typing import Optional, Sequence

import torch


def layer_norm_unfused(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm written as a chain of elementwise ops and reductions.

    Every step launches its own kernel and materialises a full-size intermediate tensor in global
    memory (centred input, squares, normalised values, scaled values, ...). This is what a model
    gets if it spells out LayerNorm by hand, and the baseline that fusion is meant to beat.
    Math is done in fp32 for half-precision inputs.
    """
    dims = tuple(range(-len(normalized_shape), 0))
    compute = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    xf = x.to(compute)
    mean = xf.mean(dim=dims, keepdim=True)
    centered = xf - mean
    var = (centered * centered).mean(dim=dims, keepdim=True)
    y = centered / torch.sqrt(var + eps)
    if weight is not None:
        y = y * weight.to(compute)
    if bias is not None:
        y = y + bias.to(compute)
    return y.to(x.dtype)
