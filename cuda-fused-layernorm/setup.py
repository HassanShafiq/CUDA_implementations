# SPDX-License-Identifier: MIT
"""Builds the ``fused_layernorm._C`` PyTorch extension.

    pip install -e . --no-build-isolation

Environment variables (all optional):
    TORCH_CUDA_ARCH_LIST   e.g. "8.6" or "7.5;8.0;8.6+PTX"; defaults to the GPUs found on this machine.
    MAX_JOBS               number of parallel compile jobs.
On Windows run from an "x64 Native Tools Command Prompt for VS 2022" with the CUDA Toolkit whose
major.minor version matches the one PyTorch was built with (``python -c "import torch; print(torch.version.cuda)"``).
"""
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

is_windows = sys.platform == "win32"

cxx_flags = ["/O2", "/std:c++17"] if is_windows else ["-O3", "-std=c++17"]
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-lineinfo",  # source-level metrics in Nsight Compute
    # NOTE: deliberately no --use_fast_math: it would trade accuracy for speed in rsqrt / division.
]

setup(
    name="fused-layernorm",
    version="1.0.0",
    description="Fused LayerNorm CUDA kernels with a PyTorch autograd integration",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        CUDAExtension(
            name="fused_layernorm._C",
            sources=[
                "csrc/torch/layernorm_ext.cpp",
                "csrc/kernels/ln_forward.cu",
                "csrc/kernels/ln_backward.cu",
            ],
            include_dirs=["csrc/include", "csrc/kernels"],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    zip_safe=False,
)
