// SPDX-License-Identifier: MIT
#include "gemm/cublas_baseline.hpp"

#include "gemm/cuda_utils.hpp"

namespace gemm {

CublasGemm::CublasGemm() {
  GEMM_CUBLAS_CHECK(cublasCreate(&handle_));
  // Default math mode: plain FP32 SGEMM (TF32 tensor-core math is not enabled),
  // so the comparison against our FP32 CUDA-core kernels is like for like.
  GEMM_CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
}

CublasGemm::~CublasGemm() {
  if (handle_ != nullptr) cublasDestroy(handle_);
}

void CublasGemm::run(const GemmArgs& g, cudaStream_t stream) const {
  GEMM_CUBLAS_CHECK(cublasSetStream(handle_, stream));
  // cuBLAS is column-major. A row-major matrix X (r x c) is the column-major
  // matrix X^T (c x r) with the same memory, so row-major C = A * B is computed
  // as column-major C^T = B^T * A^T: swap the operands and the m/n dimensions.
  GEMM_CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, g.n, g.m, g.k, &g.alpha, g.b, g.n,
                                g.a, g.k, &g.beta, g.c, g.n));
}

}  // namespace gemm
