// SPDX-License-Identifier: MIT
//
// Parallel EXCLUSIVE prefix sum of int32: a four-step ladder of block-level scans,
// composed into a full-array scan by a three-phase algorithm.
//
// Block-level kernels (each scans a tile entirely on chip and emits the tile total):
//   1 hillis_steele    O(n log n) work, log n steps, double-buffered shared memory
//   2 blelloch         O(n) work (up-sweep + down-sweep), but strided shared access -> bank conflicts
//   3 blelloch_padded  same algorithm; index padding removes the bank conflicts
//   4 warp_shuffle     4 items per thread (int4 loads), register shuffles inside warps,
//                      only two __syncthreads per tile
//
// Full-array scan (any n up to 2^30), for tile size T:
//   phase 1  scan every tile independently, write its total to sums[tile]
//   phase 2  exclusive-scan `sums` (recursively, with the same kernel)
//   phase 3  add the scanned total of preceding tiles to every element of each tile
// Phases 1 and 3 each read and write the whole array (~4n words of traffic in total,
// twice the 2n minimum). Single-pass "decoupled look-back" scans, as used by CUB,
// avoid phase 3; see docs/OPTIMIZATION_NOTES.md.
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "device_utils.cuh"
#include "prim/cuda_utils.hpp"
#include "prim/primitives.hpp"

namespace prim {
namespace {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / dev::kWarpSize;
constexpr int kWarpItems = 4;  // items per thread in the shuffle variant

// ---------------------------------------------------------------------------
// Variant 1: Hillis-Steele. Tile = kBlock elements.
// Step d: x[i] += x[i - 2^d]. All threads are active in every step, which is
// simple and fast for small tiles, but total work is O(n log n).
// ---------------------------------------------------------------------------
__global__ void scan_hillis_steele(const int* __restrict__ in, int* __restrict__ out,
                                   int* __restrict__ sums, int n) {
  __shared__ int buf[2][kBlock];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * kBlock + tid;

  const int x = (gid < n) ? in[gid] : 0;
  int read_buf = 0;
  buf[0][tid] = x;
  __syncthreads();

  for (int offset = 1; offset < kBlock; offset <<= 1) {
    const int write_buf = read_buf ^ 1;
    buf[write_buf][tid] = buf[read_buf][tid] + ((tid >= offset) ? buf[read_buf][tid - offset] : 0);
    __syncthreads();
    read_buf = write_buf;
  }

  const int inclusive = buf[read_buf][tid];
  if (gid < n) out[gid] = inclusive - x;  // exclusive = inclusive - own value
  if (sums != nullptr && tid == kBlock - 1) sums[blockIdx.x] = inclusive;
}

// ---------------------------------------------------------------------------
// Variants 2 and 3: Blelloch work-efficient scan. Tile = 2 * kBlock elements.
// Up-sweep builds a tree of partial sums, the root is cleared, the down-sweep
// pushes prefixes back down. O(n) work, but at stride 2^d the threads access shared
// memory 2^d words apart -> bank conflicts. Padding the index by (i >> 5), one word
// per 32, shifts conflicting accesses into different banks.
// ---------------------------------------------------------------------------
template <bool kPad>
__device__ __forceinline__ int pad(int i) {
  return kPad ? (i >> 5) : 0;
}

template <bool kPad>
__global__ void scan_blelloch(const int* __restrict__ in, int* __restrict__ out,
                              int* __restrict__ sums, int n) {
  constexpr int kElems = 2 * kBlock;
  constexpr int kSize = kElems + (kPad ? kElems / 32 : 0);
  __shared__ int temp[kSize];

  const int tid = threadIdx.x;
  const int base = blockIdx.x * kElems;
  const int a = tid;           // thread loads elements tid and tid + kBlock: coalesced
  const int b = tid + kBlock;
  const int pa = a + pad<kPad>(a);
  const int pb = b + pad<kPad>(b);
  temp[pa] = (base + a < n) ? in[base + a] : 0;
  temp[pb] = (base + b < n) ? in[base + b] : 0;

  // Up-sweep (reduce): build partial sums in place.
  int offset = 1;
  for (int d = kElems >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += pad<kPad>(ai);
      bi += pad<kPad>(bi);
      temp[bi] += temp[ai];
    }
    offset <<= 1;
  }

  // The root now holds the tile total: publish it and clear it for the down-sweep.
  if (tid == 0) {
    const int last = (kElems - 1) + pad<kPad>(kElems - 1);
    if (sums != nullptr) sums[blockIdx.x] = temp[last];
    temp[last] = 0;
  }

  // Down-sweep: traverse back down, swapping and accumulating.
  for (int d = 1; d < kElems; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += pad<kPad>(ai);
      bi += pad<kPad>(bi);
      const int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  if (base + a < n) out[base + a] = temp[pa];
  if (base + b < n) out[base + b] = temp[pb];
}

// ---------------------------------------------------------------------------
// Variant 4: warp-shuffle scan. Tile = kBlock * kWarpItems elements.
//  1. each thread loads 4 consecutive ints (one 128-bit load) and scans them in registers;
//  2. the 4-item totals are scanned inside each warp with register shuffles (Kogge-Stone);
//  3. the 8 warp totals are scanned by warp 0;
//  4. every thread adds its warp-level and block-level base to its 4 results.
// Only two __syncthreads are needed, versus ~2*log2(tile) for the tree-based variants.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kBlock)
    scan_warp_shuffle(const int* __restrict__ in, int* __restrict__ out, int* __restrict__ sums, int n) {
  constexpr int kElems = kBlock * kWarpItems;
  __shared__ int warp_totals[kWarps];
  __shared__ int warp_offsets[kWarps];

  const int tid = threadIdx.x;
  const int lane = tid % dev::kWarpSize;
  const int warp = tid / dev::kWarpSize;
  const int base = blockIdx.x * kElems + tid * kWarpItems;  // first element owned by this thread

  const bool vec_in = dev::is_aligned16(in);
  const bool vec_out = dev::is_aligned16(out);
  const bool full = base + kWarpItems <= n;

  int v[kWarpItems];
  if (vec_in && full) {
    const int4 t = *reinterpret_cast<const int4*>(in + base);
    v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
  } else {
#pragma unroll
    for (int j = 0; j < kWarpItems; ++j) v[j] = (base + j < n) ? in[base + j] : 0;
  }

  // (1) exclusive scan of the thread's own items
  int local[kWarpItems];
  int running = 0;
#pragma unroll
  for (int j = 0; j < kWarpItems; ++j) {
    local[j] = running;
    running += v[j];
  }
  const int thread_total = running;

  // (2) inclusive scan of thread totals across the warp
  int inclusive = thread_total;
#pragma unroll
  for (int offset = 1; offset < dev::kWarpSize; offset <<= 1) {
    const int other = __shfl_up_sync(dev::kFullMask, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  const int warp_exclusive = inclusive - thread_total;
  if (lane == dev::kWarpSize - 1) warp_totals[warp] = inclusive;
  __syncthreads();

  // (3) warp 0 scans the warp totals
  if (warp == 0) {
    const int w = (lane < kWarps) ? warp_totals[lane] : 0;
    int w_inclusive = w;
#pragma unroll
    for (int offset = 1; offset < dev::kWarpSize; offset <<= 1) {
      const int other = __shfl_up_sync(dev::kFullMask, w_inclusive, offset);
      if (lane >= offset) w_inclusive += other;
    }
    if (lane < kWarps) warp_offsets[lane] = w_inclusive - w;
    if (lane == kWarps - 1 && sums != nullptr) sums[blockIdx.x] = w_inclusive;  // tile total
  }
  __syncthreads();

  // (4) combine and store
  const int thread_base = warp_offsets[warp] + warp_exclusive;
  int r[kWarpItems];
#pragma unroll
  for (int j = 0; j < kWarpItems; ++j) r[j] = thread_base + local[j];

  if (vec_out && full) {
    *reinterpret_cast<int4*>(out + base) = make_int4(r[0], r[1], r[2], r[3]);
  } else {
#pragma unroll
    for (int j = 0; j < kWarpItems; ++j) {
      if (base + j < n) out[base + j] = r[j];
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 3: add the scanned total of all preceding tiles to every element of a tile.
// ---------------------------------------------------------------------------
__global__ void add_block_offsets(int* __restrict__ out, const int* __restrict__ offsets, int n,
                                  int elems_per_block) {
  const int base = blockIdx.x * elems_per_block;
  const int end = (base + elems_per_block < n) ? base + elems_per_block : n;
  const int add = offsets[blockIdx.x];
  for (int i = base + threadIdx.x; i < end; i += kBlock) out[i] += add;
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
using BlockScanFn = void (*)(const int* in, int* out, int* sums, int n, int blocks, cudaStream_t stream);

void block_scan_hillis_steele(const int* in, int* out, int* sums, int n, int blocks, cudaStream_t stream) {
  scan_hillis_steele<<<blocks, kBlock, 0, stream>>>(in, out, sums, n);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void block_scan_blelloch(const int* in, int* out, int* sums, int n, int blocks, cudaStream_t stream) {
  scan_blelloch<false><<<blocks, kBlock, 0, stream>>>(in, out, sums, n);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void block_scan_blelloch_padded(const int* in, int* out, int* sums, int n, int blocks, cudaStream_t stream) {
  scan_blelloch<true><<<blocks, kBlock, 0, stream>>>(in, out, sums, n);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void block_scan_warp_shuffle(const int* in, int* out, int* sums, int n, int blocks, cudaStream_t stream) {
  scan_warp_shuffle<<<blocks, kBlock, 0, stream>>>(in, out, sums, n);
  PRIM_CUDA_CHECK(cudaGetLastError());
}

constexpr int kElemsHillisSteele = kBlock;
constexpr int kElemsBlelloch = 2 * kBlock;
constexpr int kElemsWarpShuffle = kBlock * kWarpItems;

// Per-level arrays are rounded up to a multiple of 16 ints (64 bytes) so that every
// sub-array of the scratch buffer stays 16-byte aligned for the vectorised kernel.
constexpr int round_up_16(int v) { return (v + 15) / 16 * 16; }

// Scratch layout per recursion level: [ sums | scanned sums | next level ... ].
template <int kElems>
std::size_t scratch_elems(int n) {
  std::size_t total = 0;
  int length = n;
  for (;;) {
    const int blocks = ceil_div(length, kElems);
    if (blocks <= 1) break;
    total += 2 * static_cast<std::size_t>(round_up_16(blocks));
    length = blocks;
  }
  return total;
}

template <int kElems, BlockScanFn BlockScan>
void scan_level(const int* in, int* out, int n, int* scratch, cudaStream_t stream) {
  const int blocks = ceil_div(n, kElems);
  if (blocks == 1) {  // everything fits in one tile: no sums, no offsets
    BlockScan(in, out, nullptr, n, 1, stream);
    return;
  }
  int* sums = scratch;
  int* scanned_sums = scratch + round_up_16(blocks);
  int* next_level = scanned_sums + round_up_16(blocks);

  BlockScan(in, out, sums, n, blocks, stream);                                // phase 1
  scan_level<kElems, BlockScan>(sums, scanned_sums, blocks, next_level, stream);  // phase 2
  add_block_offsets<<<blocks, kBlock, 0, stream>>>(out, scanned_sums, n, kElems);  // phase 3
  PRIM_CUDA_CHECK(cudaGetLastError());
}

template <int kElems, BlockScanFn BlockScan>
void run_scan(const int* in, int* out, int n, int* scratch, cudaStream_t stream) {
  check_count(n, "n");
  if (n == 0) return;
  if (ceil_div(n, kElems) > 1 && scratch == nullptr) {
    throw std::invalid_argument("scan: scratch buffer required for this n");
  }
  scan_level<kElems, BlockScan>(in, out, n, scratch, stream);
}

}  // namespace

const std::vector<ScanKernel>& scan_registry() {
  static const std::vector<ScanKernel> registry = {
      {1, "hillis_steele", "O(n log n) work, double-buffered",
       scratch_elems<kElemsHillisSteele>, run_scan<kElemsHillisSteele, block_scan_hillis_steele>},
      {2, "blelloch", "work-efficient up/down-sweep, bank conflicts",
       scratch_elems<kElemsBlelloch>, run_scan<kElemsBlelloch, block_scan_blelloch>},
      {3, "blelloch_padded", "work-efficient, conflict-free via index padding",
       scratch_elems<kElemsBlelloch>, run_scan<kElemsBlelloch, block_scan_blelloch_padded>},
      {4, "warp_shuffle", "4 items/thread, int4 loads, warp shuffles",
       scratch_elems<kElemsWarpShuffle>, run_scan<kElemsWarpShuffle, block_scan_warp_shuffle>},
  };
  return registry;
}

}  // namespace prim
