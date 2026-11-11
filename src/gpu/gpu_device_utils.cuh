#pragma once

#include <cassert>
#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
/// Threads per block of every kernel launched through the helpers below, and the size of every __shared__ array
/// indexed by threadIdx.x.
inline constexpr unsigned int block_size = 256;

/// Index of the calling thread in a one-dimensional grid, computed in size_t so that it cannot overflow.
__device__ inline std::size_t global_index()
{
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

/// Launches `kernel` with `grid` blocks of `block` threads on `stream`, then checks the launch.
template <typename... Params, typename... Args>
void launch(void (*kernel)(Params...), unsigned int grid, unsigned int block, cudaStream_t stream, Args &&...args)
{
  kernel<<<grid, block, 0, stream>>>(std::forward<Args>(args)...);
  CHECK_LAST_GPU_ERROR();
}

/// One thread per element of [0, n).
template <typename... Params, typename... Args>
void launch_per_element(void (*kernel)(Params...), std::size_t n, cudaStream_t stream, Args &&...args)
{
  assert(n > 0);
  launch(kernel, static_cast<unsigned int>((n + block_size - 1) / block_size), block_size, stream,
         std::forward<Args>(args)...);
}

/// One block of block_size threads per island.
template <typename... Params, typename... Args>
void launch_per_island(void (*kernel)(Params...), unsigned int n_islands, cudaStream_t stream, Args &&...args)
{
  assert(n_islands > 0);
  launch(kernel, n_islands, block_size, stream, std::forward<Args>(args)...);
}
} // namespace breifast::gpu
