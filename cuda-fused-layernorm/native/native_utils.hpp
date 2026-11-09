// SPDX-License-Identifier: MIT
//
// Host-side helpers for the native (PyTorch-free) tests and benchmark: RAII device buffers,
// event timer, and host conversions between double and the storage types.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>
#include <utility>

#include "layernorm.h"

namespace native {

/// Owning, move-only device allocation of `count` elements of T.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ == 0) return;
    void* raw = nullptr;
    LN_CUDA_CHECK(cudaMalloc(&raw, count_ * sizeof(T)));
    ptr_ = static_cast<T*>(raw);
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept
      : ptr_(std::exchange(o.ptr_, nullptr)), count_(std::exchange(o.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = std::exchange(o.ptr_, nullptr);
      count_ = std::exchange(o.count_, 0);
    }
    return *this;
  }

  T* get() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return count_; }

  void upload(const T* src, std::size_t count) {
    LN_CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
  }
  void download(T* dst, std::size_t count) const {
    LN_CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

/// GPU timer based on CUDA events.
class GpuTimer {
 public:
  GpuTimer() {
    LN_CUDA_CHECK(cudaEventCreate(&start_));
    LN_CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start(cudaStream_t s = nullptr) { LN_CUDA_CHECK(cudaEventRecord(start_, s)); }
  float stop(cudaStream_t s = nullptr) {
    LN_CUDA_CHECK(cudaEventRecord(stop_, s));
    LN_CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    LN_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

// ---- storage type <-> double ------------------------------------------------------------------
inline double to_double(float v) { return v; }
inline double to_double(double v) { return v; }
inline double to_double(__half v) { return __half2float(v); }
inline double to_double(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
T from_double(double v);
template <>
inline float from_double<float>(double v) { return static_cast<float>(v); }
template <>
inline double from_double<double>(double v) { return v; }
template <>
inline __half from_double<__half>(double v) { return __float2half(static_cast<float>(v)); }
template <>
inline __nv_bfloat16 from_double<__nv_bfloat16>(double v) { return __float2bfloat16(static_cast<float>(v)); }

/// Type of the per-row statistics (mean / rstd) the kernels store for storage type T.
template <typename T>
using stat_t = std::conditional_t<std::is_same_v<T, double>, double, float>;

template <typename T> struct DTypeOf;
template <> struct DTypeOf<float> { static constexpr ln::DType value = ln::DType::kFloat32; };
template <> struct DTypeOf<double> { static constexpr ln::DType value = ln::DType::kFloat64; };
template <> struct DTypeOf<__half> { static constexpr ln::DType value = ln::DType::kFloat16; };
template <> struct DTypeOf<__nv_bfloat16> { static constexpr ln::DType value = ln::DType::kBFloat16; };

}  // namespace native
