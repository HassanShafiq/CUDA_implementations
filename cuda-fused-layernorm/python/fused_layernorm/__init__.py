# SPDX-License-Identifier: MIT
"""Fused LayerNorm CUDA kernels with a PyTorch autograd integration.

Public API
----------
fused_layer_norm  functional interface, drop-in for ``torch.nn.functional.layer_norm``
FusedLayerNorm    ``nn.Module``, drop-in for ``torch.nn.LayerNorm``
forward_kernels   list of the forward-kernel ladder as ``(id, name, summary)``
"""
from __future__ import annotations

try:
    from . import _C
except ImportError as exc:  # pragma: no cover - depends on the build environment
    raise ImportError(
        "The compiled extension 'fused_layernorm._C' was not found. Build it from the repository "
        "root with:  pip install -e . --no-build-isolation  (see README.md for Windows notes)."
    ) from exc

from .functional import fused_layer_norm
from .module import FusedLayerNorm

__version__ = "1.0.0"
__all__ = ["fused_layer_norm", "FusedLayerNorm", "forward_kernels", "kernel_supported"]


def forward_kernels():
    """Returns the forward-kernel ladder as a list of ``(id, name, summary)``; id 0 means auto."""
    return list(_C.forward_kernels())


def kernel_supported(kernel_id, x, weight=None, bias=None):
    """Whether forward kernel ``kernel_id`` can run on these tensors (shape, dtype, alignment).

    The last dimension of ``x`` is taken as the normalised dimension.
    """
    cols = x.shape[-1]
    return _C.kernel_supported(kernel_id, x.contiguous().view(x.numel() // cols, cols), weight, bias)
