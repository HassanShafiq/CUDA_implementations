# SPDX-License-Identifier: MIT
"""``nn.Module`` wrapper: a drop-in replacement for ``torch.nn.LayerNorm`` backed by fused kernels."""
from __future__ import annotations

import math
from typing import Optional, Union, Sequence

import torch
from torch import nn

from .functional import fused_layer_norm


class FusedLayerNorm(nn.Module):
    """Same constructor semantics and parameter names (``weight``, ``bias``) as ``torch.nn.LayerNorm``.

    A state dict saved from ``nn.LayerNorm`` loads into this module and vice versa.

    Args:
        normalized_shape: int or sequence of ints, the trailing dimensions to normalise over.
        eps: variance epsilon.
        elementwise_affine: if True, learn ``weight`` (ones) and, with ``bias=True``, ``bias`` (zeros).
        bias: whether to learn an additive bias (only meaningful with ``elementwise_affine``).
        device, dtype: factory kwargs for the parameters.
        kernel_id: forward kernel selection; 0 = automatic (see ``fused_layernorm.forward_kernels``).
    """

    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        kernel_id: int = 0,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.kernel_id = kernel_id
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, **factory))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps, self.kernel_id)

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, "
            f"kernel_id={self.kernel_id}"
        )
