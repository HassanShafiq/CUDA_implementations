// SPDX-License-Identifier: MIT
//
// cuBLAS SGEMM wrapper used as the performance and correctness reference.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gemm/kernels.hpp"

namespace gemm {

class CublasGemm {
 public:
  CublasGemm();
  ~CublasGemm();

  CublasGemm(const CublasGemm&) = delete;
  CublasGemm& operator=(const CublasGemm&) = delete;

  /// Runs the same row-major problem described by `args` through cublasSgemm.
  void run(const GemmArgs& args, cudaStream_t stream) const;

 private:
  cublasHandle_t handle_ = nullptr;
};

}  // namespace gemm
