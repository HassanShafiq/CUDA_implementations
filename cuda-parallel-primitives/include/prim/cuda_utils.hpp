// SPDX-License-Identifier: MIT
//
// Host-side CUDA helpers: error checking and RAII wrappers.
//
//  * Failures throw `CudaError` (never exit()), so RAII cleans up and tests can
//    report instead of aborting.
//  * Device memory and events are owned by objects whose destructors release them;
//    there are no raw cudaMalloc/cudaFree pairs elsewhere in the code base.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace prim {

/// Thrown when a CUDA runtime call fails.
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

}  // namespace detail
}  // namespace prim

#define PRIM_CUDA_CHECK(expr) ::prim::detail::check_cuda((expr), #expr, __FILE__, __LINE__)

namespace prim {

/// Owning, move-only handle to a device allocation of `count` elements of T.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ == 0) return;
    void* raw = nullptr;
    PRIM_CUDA_CHECK(cudaMalloc(&raw, count_ * sizeof(T)));
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

  /// Copies `count` elements from host memory into the buffer, starting at element `offset`.
  void copy_from_host(const T* src, std::size_t count, std::size_t offset = 0) {
    check_range(offset, count);
    PRIM_CUDA_CHECK(cudaMemcpy(ptr_ + offset, src, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  /// Copies `count` elements to host memory, starting at element `offset` of the buffer.
  void copy_to_host(T* dst, std::size_t count, std::size_t offset = 0) const {
    check_range(offset, count);
    PRIM_CUDA_CHECK(cudaMemcpy(dst, ptr_ + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

 private:
  void check_range(std::size_t offset, std::size_t count) const {
    if (offset + count > count_) throw CudaError("DeviceBuffer: copy exceeds allocation size");
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
    PRIM_CUDA_CHECK(cudaEventCreate(&start_));
    PRIM_CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start(cudaStream_t stream = nullptr) { PRIM_CUDA_CHECK(cudaEventRecord(start_, stream)); }

  /// Records the stop event, waits for it, and returns elapsed milliseconds.
  float stop(cudaStream_t stream = nullptr) {
    PRIM_CUDA_CHECK(cudaEventRecord(stop_, stream));
    PRIM_CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    PRIM_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace prim
