// SPDX-License-Identifier: MIT
//
// Parallel sum reduction: a five-step optimisation ladder.
//
// Reduction does one add per 4 bytes read (0.25 FLOP/byte), so it is purely memory-bound:
// the goal of every step is to get closer to the GPU's achievable memory bandwidth
// by removing everything that is not "load data" (idle threads, divergence, bank
// conflicts, synchronisation, launch overhead).
//
//   1 interleaved_divergent   tid % (2*s) == 0              -> warp divergence, slow modulo
//   2 interleaved_strided     index = 2*s*tid               -> no divergence, shared-memory bank conflicts
//   3 sequential              tid < s: a[tid] += a[tid + s] -> conflict-free
//   4 first_add_on_load       each thread loads and adds two elements -> half the blocks, no idle half
//   5 warp_shuffle            grid-stride loop + float4 loads + warp shuffles -> few blocks, few syncs
//
// A single kernel launch can only reduce a block's worth of data to one value, so
// every variant runs as repeated passes: n -> per-block partial sums -> ... -> 1 value.
// Floating-point addition is not associative, so results differ from variant to variant
// in the last bits; tests use integer-valued floats, which sum exactly in any order.
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "device_utils.cuh"
#include "prim/cuda_utils.hpp"
#include "prim/primitives.hpp"

namespace prim {
namespace {

constexpr int kBlock = 256;                    // threads per block (power of two)
constexpr int kWarps = kBlock / dev::kWarpSize;
constexpr int kMaxGridStrideBlocks = 1024;     // cap for the persistent-style variant
constexpr int kItemsPerThread = 16;            // target work per thread in variant 5

// ---------------------------------------------------------------------------
// Variant 1: interleaved addressing with a divergent branch.
// In step s only threads with tid % (2s) == 0 work. Active threads are scattered
// across every warp, so warps stay busy while most of their lanes are idle, and
// the % operator is expensive.
// ---------------------------------------------------------------------------
__global__ void reduce_interleaved_divergent(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float sdata[kBlock];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * kBlock + tid;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  for (int s = 1; s < kBlock; s <<= 1) {
    if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Variant 2: interleaved addressing, contiguous active threads.
// Thread t works on element 2*s*t, so the active threads are the first ones and
// whole warps retire early (no divergence). But consecutive threads now access
// shared memory with a stride of 2*s words -> up to 32-way bank conflicts.
// ---------------------------------------------------------------------------
__global__ void reduce_interleaved_strided(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float sdata[kBlock];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * kBlock + tid;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  for (int s = 1; s < kBlock; s <<= 1) {
    const int index = 2 * s * tid;
    if (index < kBlock) sdata[index] += sdata[index + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Variant 3: sequential addressing.
// Thread t adds element t + s, with s halving each step. Active threads are
// contiguous (no divergence) and read/write consecutive words (no bank conflicts).
// Remaining inefficiency: half of the threads do nothing in the very first step.
// ---------------------------------------------------------------------------
__global__ void reduce_sequential(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float sdata[kBlock];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * kBlock + tid;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Variant 4: first add during load.
// Each block covers 2 * kBlock elements: every thread loads two values and adds
// them before touching shared memory. Half as many blocks, and no thread is idle
// in what used to be the first reduction step.
// ---------------------------------------------------------------------------
__global__ void reduce_first_add_on_load(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float sdata[kBlock];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * (2 * kBlock) + tid;
  float v = 0.0f;
  if (i < n) v = in[i];
  if (i + kBlock < n) v += in[i + kBlock];
  sdata[tid] = v;
  __syncthreads();

  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Variant 5: grid-stride loop, float4 loads, warp-shuffle reduction.
//  * A bounded number of blocks (<= 1024) each loops over the input with a grid
//    stride, so every thread accumulates ~16 elements in a register: fewer blocks to
//    schedule, fewer syncs per element, and independent loads in flight.
//  * 128-bit loads (when `in` is 16-byte aligned; otherwise a scalar loop is used).
//  * Within a warp, values are combined with register shuffles (no shared memory,
//    no __syncthreads); shared memory is only used to combine the 8 warp results.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kBlock)
    reduce_warp_shuffle(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float warp_sums[kWarps];
  const int tid = threadIdx.x;
  const int lane = tid % dev::kWarpSize;
  const int warp = tid / dev::kWarpSize;
  const int gid = blockIdx.x * kBlock + tid;
  const int stride = gridDim.x * kBlock;

  float sum = 0.0f;
  int scalar_start = 0;
  if (dev::is_aligned16(in)) {
    const int n4 = n >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (int i = gid; i < n4; i += stride) {
      const float4 v = in4[i];
      sum += (v.x + v.y) + (v.z + v.w);
    }
    scalar_start = n4 << 2;  // the (at most 3) elements that do not fill a float4
  }
  for (int i = scalar_start + gid; i < n; i += stride) sum += in[i];

  sum = dev::warp_reduce_sum(sum);
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();

  if (warp == 0) {
    float v = (lane < kWarps) ? warp_sums[lane] : 0.0f;
    v = dev::warp_reduce_sum(v);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

// ---------------------------------------------------------------------------
// Host side: one pass = one kernel launch; the driver repeats passes until a
// single value remains.
// ---------------------------------------------------------------------------
using BlocksFn = int (*)(int n);
using PassFn = void (*)(const float* in, float* out, int n, int blocks, cudaStream_t stream);

int blocks_one_per_thread(int n) { return ceil_div(n, kBlock); }
int blocks_two_per_thread(int n) { return ceil_div(n, 2 * kBlock); }
int blocks_grid_stride(int n) {
  const int wanted = ceil_div(n, kBlock * kItemsPerThread);
  return wanted < kMaxGridStrideBlocks ? wanted : kMaxGridStrideBlocks;
}

void pass_interleaved_divergent(const float* in, float* out, int n, int blocks, cudaStream_t stream) {
  reduce_interleaved_divergent<<<blocks, kBlock, 0, stream>>>(in, out, n);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void pass_interleaved_strided(const float* in, float* out, int n, int blocks, cudaStream_t stream) {
  reduce_interleaved_strided<<<blocks, kBlock, 0, stream>>>(in, out, n);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void pass_sequential(const float* in, float* out, int n, int blocks, cudaStream_t stream) {
  reduce_sequential<<<blocks, kBlock, 0, stream>>>(in, out, n);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void pass_first_add_on_load(const float* in, float* out, int n, int blocks, cudaStream_t stream) {
  reduce_first_add_on_load<<<blocks, kBlock, 0, stream>>>(in, out, n);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void pass_warp_shuffle(const float* in, float* out, int n, int blocks, cudaStream_t stream) {
  reduce_warp_shuffle<<<blocks, kBlock, 0, stream>>>(in, out, n);
  PRIM_CUDA_CHECK(cudaGetLastError());
}

// Scratch holds two ping-pong buffers of `first_pass_blocks` floats; none is needed
// when a single block suffices.
template <BlocksFn Blocks>
std::size_t scratch_elems(int n) {
  if (n <= 0) return 0;
  const int first = Blocks(n);
  return first > 1 ? 2 * static_cast<std::size_t>(first) : 0;
}

template <BlocksFn Blocks, PassFn Pass>
void run_reduce(const float* in, float* out, int n, float* scratch, cudaStream_t stream) {
  check_count(n, "n");
  if (n == 0) {
    PRIM_CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
    return;
  }
  const int first = Blocks(n);
  if (first > 1 && scratch == nullptr) throw std::invalid_argument("reduce: scratch buffer required for this n");

  float* buffers[2] = {scratch, scratch + first};
  const float* current = in;
  int current_n = n;
  int which = 0;
  for (;;) {
    const int blocks = Blocks(current_n);
    if (blocks == 1) {
      Pass(current, out, current_n, 1, stream);  // last pass writes the final value
      return;
    }
    float* destination = buffers[which];
    Pass(current, destination, current_n, blocks, stream);
    current = destination;
    current_n = blocks;
    which ^= 1;  // read what was just written, write into the other buffer
  }
}

}  // namespace

const std::vector<ReduceKernel>& reduce_registry() {
  static const std::vector<ReduceKernel> registry = {
      {1, "interleaved_divergent", "interleaved addressing, divergent branches",
       scratch_elems<blocks_one_per_thread>, run_reduce<blocks_one_per_thread, pass_interleaved_divergent>},
      {2, "interleaved_strided", "contiguous active threads, shared-memory bank conflicts",
       scratch_elems<blocks_one_per_thread>, run_reduce<blocks_one_per_thread, pass_interleaved_strided>},
      {3, "sequential", "sequential addressing, conflict-free",
       scratch_elems<blocks_one_per_thread>, run_reduce<blocks_one_per_thread, pass_sequential>},
      {4, "first_add_on_load", "two elements per thread added during the load",
       scratch_elems<blocks_two_per_thread>, run_reduce<blocks_two_per_thread, pass_first_add_on_load>},
      {5, "warp_shuffle", "grid-stride loop, float4 loads, warp shuffles",
       scratch_elems<blocks_grid_stride>, run_reduce<blocks_grid_stride, pass_warp_shuffle>},
  };
  return registry;
}

}  // namespace prim
