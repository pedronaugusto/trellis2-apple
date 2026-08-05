"""The auto-detected dense-attention backend must actually work on this host.

Regression test. `trellis2/modules/attention/config.py` used to hardcode
BACKEND = 'flash_attn'. flash-attn is a CUDA-only extension and cannot be
installed on Apple Silicon at all, so on macOS the very first transformer
block of the sparse-structure flow raised ModuleNotFoundError. It went
unnoticed because the only caller that set the env var was `api_server.py`
(`os.environ.setdefault("ATTN_BACKEND", "sdpa")`) — so the server worked
while the README's own minimal example could never have run on a Mac.

This asserts the *auto-detected* default, so ATTN_BACKEND is cleared before
the config module is imported. It deliberately does not hardcode which
backend is expected: it checks that whatever the platform selects is both
importable and numerically correct, which is the property that was violated.
"""

import os

os.environ.pop("ATTN_BACKEND", None)
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import pytest
import torch

from trellis2.modules.attention import config
from trellis2.modules.attention.full_attn import (
    scaled_dot_product_attention,
    _naive_sdpa,
)


def _device_and_dtype():
    if torch.cuda.is_available():
        # flash-attn family is half-precision only.
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def test_default_backend_is_importable_and_correct():
    device, dtype = _device_and_dtype()
    torch.manual_seed(0)

    N, L, H, C = 2, 16, 4, 32
    q, k, v = (torch.randn(N, L, H, C, device=device, dtype=dtype) for _ in range(3))

    # Fails with ModuleNotFoundError if the platform default names a backend
    # whose extension isn't installable here — the original bug.
    out = scaled_dot_product_attention(q, k, v)

    assert out.shape == (N, L, H, C), f"got {tuple(out.shape)}"
    assert out.device.type == device
    assert torch.isfinite(out).all(), f"non-finite output from backend {config.BACKEND!r}"

    ref = _naive_sdpa(q.float(), k.float(), v.float())
    tol = 2e-2 if dtype == torch.float16 else 1e-4
    diff = (out.float() - ref).abs().max().item()
    assert diff <= tol, f"backend {config.BACKEND!r} disagrees with naive sdpa: {diff:.3e} > {tol:.0e}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon only")
def test_apple_silicon_does_not_default_to_flash_attn():
    assert config.BACKEND != "flash_attn", (
        "flash-attn is CUDA-only and cannot be installed on Apple Silicon; "
        "the platform default must not select it"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
