# SPDX-License-Identifier: MIT
"""Functional interface and autograd integration of the fused LayerNorm kernels."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch.autograd.function import once_differentiable

from . import _C

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.float64)

Shape = Union[int, Sequence[int], torch.Size]


def _as_shape(normalized_shape: Shape) -> Tuple[int, ...]:
    if isinstance(normalized_shape, int):
        return (normalized_shape,)
    return tuple(int(s) for s in normalized_shape)


def _validate(x: torch.Tensor, shape: Tuple[int, ...], weight, bias, kernel_id: int) -> None:
    if not x.is_cuda:
        raise ValueError("fused_layer_norm requires a CUDA tensor")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {x.dtype}; supported: {[str(d) for d in _SUPPORTED_DTYPES]}")
    if len(shape) == 0 or any(s <= 0 for s in shape):
        raise ValueError(f"normalized_shape must be non-empty and positive, got {shape}")
    if x.dim() < len(shape) or tuple(x.shape[-len(shape):]) != shape:
        raise ValueError(f"input of shape {tuple(x.shape)} cannot be normalized over the last dims {shape}")
    for name, p in (("weight", weight), ("bias", bias)):
        if p is None:
            continue
        if tuple(p.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(p.shape)}")
        if p.dtype != x.dtype:
            raise ValueError(f"{name} dtype ({p.dtype}) must match the input dtype ({x.dtype})")
        if p.device != x.device:
            raise ValueError(f"{name} must be on the same device as the input")
    if kernel_id < 0:
        raise ValueError(f"kernel_id must be >= 0 (0 = automatic), got {kernel_id}")


class _FusedLayerNormFunction(torch.autograd.Function):
    """y = (x - mean) * rstd * weight + bias over the trailing ``normalized_shape`` dimensions.

    Forward saves only ``x``, ``weight`` and the per-row ``mean`` / ``rstd`` vectors (O(rows) extra
    memory); backward reuses them instead of recomputing the statistics.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, normalized_shape, eps, kernel_id):
        cols = math.prod(normalized_shape)
        rows = x.numel() // cols  # explicit (not -1) so that empty batches work
        x2d = x.contiguous().view(rows, cols)
        w = weight.contiguous().view(-1) if weight is not None else None
        b = bias.contiguous().view(-1) if bias is not None else None

        y2d, mean, rstd = _C.forward(x2d, w, b, float(eps), int(kernel_id))

        ctx.save_for_backward(x2d, w, mean, rstd)
        ctx.x_shape = x.shape
        ctx.param_shape = tuple(normalized_shape)
        return y2d.view(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x2d, w, mean, rstd = ctx.saved_tensors
        need_dx, need_dw, need_db = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        need_params = bool(need_dw or need_db)

        dy2d = grad_output.contiguous().view(x2d.shape)
        dx, dgamma, dbeta = _C.backward(dy2d, x2d, w, mean, rstd, need_params)

        return (
            dx.view(ctx.x_shape) if need_dx else None,
            dgamma.view(ctx.param_shape) if need_dw else None,
            dbeta.view(ctx.param_shape) if need_db else None,
            None,  # normalized_shape
            None,  # eps
            None,  # kernel_id
        )


def fused_layer_norm(
    x: torch.Tensor,
    normalized_shape: Shape,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    kernel_id: int = 0,
) -> torch.Tensor:
    """Layer normalisation over the last ``len(normalized_shape)`` dimensions of ``x``.

    Args:
        x: CUDA tensor (float32, float16, bfloat16 or float64) of shape ``[..., *normalized_shape]``.
        normalized_shape: trailing shape to normalise over (an int is treated as a 1-tuple).
        weight, bias: optional affine parameters of shape ``normalized_shape`` and the same dtype as ``x``.
        eps: value added to the variance for numerical stability.
        kernel_id: forward kernel to use; 0 (default) selects the fastest one that supports the input.
            See :func:`fused_layernorm.forward_kernels`.

    Returns:
        A tensor with the same shape and dtype as ``x``. Statistics and accumulation are done in fp32
        (fp64 for double inputs). Supports autograd (first-order only).
    """
    shape = _as_shape(normalized_shape)
    _validate(x, shape, weight, bias, int(kernel_id))
    return _FusedLayerNormFunction.apply(x, weight, bias, shape, eps, int(kernel_id))
