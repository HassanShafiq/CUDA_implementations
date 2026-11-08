// SPDX-License-Identifier: MIT
//
// Central list of kernels so the benchmark and the tests iterate over exactly
// the same set, in optimisation order.
#include <cstdint>

#include "gemm/kernels.hpp"

namespace gemm {
namespace {

bool supports_any_shape(const GemmArgs&) { return true; }

bool is_aligned_16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// float4 accesses need n and k divisible by 4 and 16-byte aligned base pointers.
bool supports_vectorized(const GemmArgs& g) {
  return g.n % 4 == 0 && g.k % 4 == 0 && is_aligned_16(g.a) && is_aligned_16(g.b) && is_aligned_16(g.c);
}

}  // namespace

const std::vector<KernelInfo>& kernel_registry() {
  static const std::vector<KernelInfo> registry = {
      {1, "naive", "one thread per C element, uncoalesced access", launch_sgemm_naive, supports_any_shape},
      {2, "coalesced", "thread mapping fixed for coalesced global access", launch_sgemm_coalesced,
       supports_any_shape},
      {3, "smem_tiled", "32x32 shared-memory tiling", launch_sgemm_smem_tiled, supports_any_shape},
      {4, "blocktile_1d", "1D register blocking (8 outputs/thread)", launch_sgemm_blocktile_1d,
       supports_any_shape},
      {5, "regtile_2d", "2D register tiling (8x8 outputs/thread), conflict-free smem",
       launch_sgemm_regtile_2d, supports_any_shape},
      {6, "vectorized", "float4 global/shared accesses", launch_sgemm_vectorized, supports_vectorized},
  };
  return registry;
}

}  // namespace gemm
