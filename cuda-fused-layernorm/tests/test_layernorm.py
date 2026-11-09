# SPDX-License-Identifier: MIT
"""Tests of the PyTorch integration (autograd, module, dtypes, kernel ladder).

Reference for forward and backward is ``torch.nn.functional.layer_norm`` evaluated in fp32 (fp64 for
double inputs) on exactly the same (already quantised) inputs, so the only differences allowed are the
rounding of the fused kernel's output to the storage dtype and fp32 accumulation order.

Run:  pytest -v          (needs a CUDA GPU and a built extension: pip install -e . --no-build-isolation)
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("fused_layernorm", reason="extension not built: pip install -e . --no-build-isolation")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from fused_layernorm import FusedLayerNorm, forward_kernels, fused_layer_norm, kernel_supported  # noqa: E402
from fused_layernorm.reference import layer_norm_unfused  # noqa: E402

DEVICE = "cuda"
DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
KERNEL_IDS = [0] + [k[0] for k in forward_kernels()]

# (atol, rtol) of the forward output / of gradients, per storage dtype.
FWD_TOL = {torch.float32: (5e-5, 1e-4), torch.float16: (4e-3, 2e-3), torch.bfloat16: (3e-2, 1e-2), torch.float64: (1e-8, 1e-8)}
BWD_TOL = {torch.float32: (2e-4, 1e-3), torch.float16: (1.5e-2, 1.5e-2), torch.bfloat16: (8e-2, 4e-2), torch.float64: (1e-8, 1e-8)}

# Shapes straddle warp / 16-byte-chunk / register-cache capacity boundaries (fp32: 1024 and 8192 columns).
SHAPES = [(1, 1), (4, 7), (8, 31), (8, 128), (16, 768), (5, 1000), (3, 1025), (2, 4096), (2, 8192), (2, 8200), (300, 64)]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1234)


def make(shape, dtype, affine=True, bias=True, mean=3.0):
    rows, cols = shape
    x = torch.randn(rows, cols, device=DEVICE, dtype=dtype) * 2 + mean
    w = (torch.rand(cols, device=DEVICE, dtype=dtype) + 0.5) if affine else None
    b = torch.randn(cols, device=DEVICE, dtype=dtype) if (affine and bias) else None
    return x, w, b


def reference(x, shape, w, b, eps=1e-5):
    wide = torch.float64 if x.dtype == torch.float64 else torch.float32
    return F.layer_norm(x.to(wide), shape, None if w is None else w.to(wide), None if b is None else b.to(wide), eps)


def close(actual, expected, tol):
    torch.testing.assert_close(actual.to(expected.dtype), expected, atol=tol[0], rtol=tol[1])


# --------------------------------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kernel_id", KERNEL_IDS)
def test_forward_matches_reference(dtype, shape, kernel_id):
    x, w, b = make(shape, dtype)
    if not kernel_supported(kernel_id, x, w, b):
        pytest.skip("kernel does not support this shape / alignment")
    y = fused_layer_norm(x, (shape[1],), w, b, 1e-5, kernel_id)
    assert y.dtype == dtype and y.shape == x.shape
    close(y, reference(x, (shape[1],), w, b), FWD_TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("affine,bias", [(False, False), (True, False)], ids=["no-affine", "weight-only"])
def test_forward_optional_parameters(dtype, affine, bias):
    x, w, b = make((8, 256), dtype, affine=affine, bias=bias)
    close(fused_layer_norm(x, (256,), w, b), reference(x, (256,), w, b), FWD_TOL[dtype])


def test_multidimensional_normalized_shape_and_leading_dims():
    x = torch.randn(2, 3, 4, 8, device=DEVICE) * 2 + 1
    w = torch.rand(4, 8, device=DEVICE) + 0.5
    b = torch.randn(4, 8, device=DEVICE)
    y = fused_layer_norm(x, (4, 8), w, b)
    close(y, F.layer_norm(x, (4, 8), w, b), FWD_TOL[torch.float32])


def test_non_contiguous_input():
    base = torch.randn(64, 512, device=DEVICE)
    for x in (base[:, :256], base.t()[:256, :].t()):  # strided view, transposed view
        assert not x.is_contiguous()
        close(fused_layer_norm(x, (x.shape[-1],)), F.layer_norm(x, (x.shape[-1],)), FWD_TOL[torch.float32])


def test_large_mean_is_numerically_stable():
    # |mean| >> std breaks E[x^2] - E[x]^2; Welford / two-pass statistics must not.
    x = torch.randn(32, 1024, device=DEVICE) + 100.0
    expected = F.layer_norm(x.double(), (1024,)).float()
    for kernel_id in KERNEL_IDS:
        if kernel_supported(kernel_id, x):
            y = fused_layer_norm(x, (1024,), None, None, 1e-5, kernel_id)
            torch.testing.assert_close(y, expected, atol=1e-4, rtol=1e-4)


def test_matches_unfused_composition():
    x, w, b = make((64, 768), torch.float16)
    close(fused_layer_norm(x, (768,), w, b), layer_norm_unfused(x, (768,), w, b).float(), FWD_TOL[torch.float16])


def test_empty_batch():
    x = torch.randn(0, 64, device=DEVICE, requires_grad=True)
    w = torch.ones(64, device=DEVICE, requires_grad=True)
    b = torch.zeros(64, device=DEVICE, requires_grad=True)
    y = fused_layer_norm(x, (64,), w, b)
    assert y.shape == (0, 64)
    y.sum().backward()
    assert x.grad.shape == (0, 64)
    assert torch.count_nonzero(w.grad) == 0 and torch.count_nonzero(b.grad) == 0


# --------------------------------------------------------------------------------------------------
# Backward
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 1), (4, 7), (16, 768), (3, 1025), (2, 4096), (300, 64), (600, 100)])
def test_backward_matches_autograd(dtype, shape):
    x, w, b = make(shape, dtype)
    g = torch.randn_like(x)
    xr, wr, br = (t.detach().to(torch.float64 if dtype == torch.float64 else torch.float32).requires_grad_() for t in (x, w, b))
    x.requires_grad_(), w.requires_grad_(), b.requires_grad_()

    fused_layer_norm(x, (shape[1],), w, b).backward(g)
    F.layer_norm(xr, (shape[1],), wr, br, 1e-5).backward(g.to(xr.dtype))

    close(x.grad, xr.grad, BWD_TOL[dtype])
    close(w.grad, wr.grad, BWD_TOL[dtype])
    close(b.grad, br.grad, BWD_TOL[dtype])
    assert x.grad.dtype == dtype and w.grad.dtype == dtype


def test_backward_without_affine_and_with_frozen_parameters():
    x, w, b = make((32, 200), torch.float32)
    x.requires_grad_()
    fused_layer_norm(x, (200,)).sum().backward()  # no weight / bias at all
    assert x.grad is not None

    w.requires_grad_(False), b.requires_grad_(False)
    x2 = x.detach().clone().requires_grad_()
    fused_layer_norm(x2, (200,), w, b).sum().backward()  # frozen parameters: only dx
    assert w.grad is None and b.grad is None and x2.grad is not None


def test_gradcheck_float64():
    x = torch.randn(3, 37, device=DEVICE, dtype=torch.float64, requires_grad=True)
    w = torch.rand(37, device=DEVICE, dtype=torch.float64, requires_grad=True)
    b = torch.randn(37, device=DEVICE, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, w, b: fused_layer_norm(x, (37,), w, b), (x, w, b), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_backward_is_bitwise_deterministic():
    x, w, b = make((777, 1000), torch.float32)
    g = torch.randn_like(x)
    grads = []
    for _ in range(3):
        xi, wi, bi = (t.detach().clone().requires_grad_() for t in (x, w, b))
        fused_layer_norm(xi, (1000,), wi, bi).backward(g)
        grads.append((xi.grad, wi.grad, bi.grad))
    for other in grads[1:]:
        assert all(torch.equal(a, c) for a, c in zip(grads[0], other))


def test_double_backward_is_rejected():
    x = torch.randn(4, 16, device=DEVICE, requires_grad=True)
    (gx,) = torch.autograd.grad(fused_layer_norm(x, (16,)).sum(), x, create_graph=True)
    with pytest.raises(RuntimeError):
        gx.sum().backward()


# --------------------------------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bias", [True, False])
def test_module_is_a_drop_in_replacement(bias):
    ref = torch.nn.LayerNorm(128, bias=bias).to(DEVICE)
    mod = FusedLayerNorm(128, bias=bias).to(DEVICE)
    with torch.no_grad():  # randomise parameters, then check the state dicts are interchangeable
        ref.weight.uniform_(0.5, 1.5)
        if bias:
            ref.bias.normal_()
    mod.load_state_dict(ref.state_dict())

    x = torch.randn(16, 128, device=DEVICE)
    g = torch.randn_like(x)
    xr, xm = x.clone().requires_grad_(), x.clone().requires_grad_()
    ref(xr).backward(g)
    mod(xm).backward(g)
    torch.testing.assert_close(mod(x), ref(x), atol=5e-5, rtol=1e-4)
    torch.testing.assert_close(xm.grad, xr.grad, atol=2e-4, rtol=1e-3)
    torch.testing.assert_close(mod.weight.grad, ref.weight.grad, atol=2e-4, rtol=1e-3)
    if bias:
        torch.testing.assert_close(mod.bias.grad, ref.bias.grad, atol=2e-4, rtol=1e-3)


def test_module_without_affine_has_no_parameters():
    assert list(FusedLayerNorm(64, elementwise_affine=False).parameters()) == []


def test_module_trains():
    torch.manual_seed(0)
    mod = FusedLayerNorm(32).to(DEVICE)
    opt = torch.optim.SGD(mod.parameters(), lr=0.1)
    x = torch.randn(64, 32, device=DEVICE)
    target = torch.randn(64, 32, device=DEVICE)
    first = None
    for _ in range(20):
        opt.zero_grad()
        loss = F.mse_loss(mod(x), target)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------
def test_invalid_arguments_raise():
    x = torch.randn(4, 16, device=DEVICE)
    with pytest.raises(ValueError):
        fused_layer_norm(x.cpu(), (16,))  # not a CUDA tensor
    with pytest.raises(ValueError):
        fused_layer_norm(x, (8,))  # shape mismatch
    with pytest.raises(ValueError):
        fused_layer_norm(x, (16,), torch.ones(16, device=DEVICE, dtype=torch.float16))  # dtype mismatch
    with pytest.raises(ValueError):
        fused_layer_norm(x.int(), (16,))  # unsupported dtype
    with pytest.raises((ValueError, RuntimeError)):
        fused_layer_norm(x, (16,), None, None, 1e-5, 99)  # unknown kernel id


def test_explicit_kernel_that_does_not_support_the_input_raises():
    x = torch.randn(4, 2000, device=DEVICE)  # too wide for the warp-per-row kernel
    assert not kernel_supported(3, x)
    with pytest.raises((ValueError, RuntimeError)):
        fused_layer_norm(x, (2000,), None, None, 1e-5, 3)
    fused_layer_norm(x, (2000,), None, None, 1e-5, 0)  # automatic selection still works
