// SPDX-License-Identifier: MIT
//
// Host-side CUDA helpers: error checking and RAII wrappers.
//
// Design notes
//  * Errors are reported with exceptions (not exit()) so callers can clean up
//    via RAII and tests can report failures instead of aborting.
//  * Every resource (device memory, events) is owned by an object whose
//    destructor releases it; there are no raw cudaMalloc/cudaFree pairs
//    elsewhere in the code base.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace gemm {

/// Thrown when a CUDA runtime or cuBLAS call fails.
class CudaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

namespace detail {

inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
  if (status != cudaSuccess) {
    throw CudaError(std::string(file) + ":" + std::to_string(line) + ": " + expr +
                    " failed: " + cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status, const char* expr, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw CudaError(std::string(file) + ":" + std::to_string(line) + ": " + expr +
                    " failed: cuBLAS status " + std::to_string(static_cast<int>(status)));
  }
}

}  // namespace detail
}  // namespace gemm

#define GEMM_CUDA_CHECK(expr) ::gemm::detail::check_cuda((expr), #expr, __FILE__, __LINE__)
#define GEMM_CUBLAS_CHECK(expr) ::gemm::detail::check_cublas((expr), #expr, __FILE__, __LINE__)

namespace gemm {

/// Owning, move-only handle to a device allocation of `count` elements of T.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ == 0) return;
    void* raw = nullptr;
    GEMM_CUDA_CHECK(cudaMalloc(&raw, count_ * sizeof(T)));
    ptr_ = static_cast<T*>(raw);
  }

  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)), count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_ = std::exchange(other.ptr_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  T* get() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return count_; }

  void copy_from_host(const T* src, std::size_t count) {
    check_range(count);
    GEMM_CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copy_to_host(T* dst, std::size_t count) const {
    check_range(count);
    GEMM_CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

 private:
  void check_range(std::size_t count) const {
    if (count > count_) throw CudaError("DeviceBuffer: copy exceeds allocation size");
  }

  void release() noexcept {
    if (ptr_ != nullptr) cudaFree(ptr_);  // destructor path: nothing useful to do on failure
    ptr_ = nullptr;
    count_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

/// Measures GPU time between two points of a stream using CUDA events.
class GpuTimer {
 public:
  GpuTimer() {
    GEMM_CUDA_CHECK(cudaEventCreate(&start_));
    GEMM_CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start(cudaStream_t stream = nullptr) { GEMM_CUDA_CHECK(cudaEventRecord(start_, stream)); }

  /// Records the stop event, waits for it, and returns elapsed milliseconds.
  float stop(cudaStream_t stream = nullptr) {
    GEMM_CUDA_CHECK(cudaEventRecord(stop_, stream));
    GEMM_CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    GEMM_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace gemm
