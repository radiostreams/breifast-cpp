#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <cub/block/block_reduce.cuh>
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

/// fminf and fmaxf as function objects, for atomic_update and block_reduce. A NaN operand yields the other one.
struct FloatMin
{
  __device__ float operator()(const float a, const float b) const { return fminf(a, b); }
};
struct FloatMax
{
  __device__ float operator()(const float a, const float b) const { return fmaxf(a, b); }
};

/// Atomically replaces `*address` by `op(value, *address)` with a compare-and-swap loop, for float operations that
/// have no hardware atomic (FloatMin, FloatMax).
template <typename Op> __device__ void atomic_update(float *const address, const float value, const Op op)
{
  int *const address_as_int = reinterpret_cast<int *>(address);
  int old = *address_as_int;
  int assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(op(value, __int_as_float(assumed))));
  } while (assumed != old);
}

/// Reduces `value` over the block with `op` and returns the result to every thread. Every thread of a block of
/// block_size threads must call it. The leading barrier makes back-to-back calls safe: a call cannot overwrite the
/// shared memory that threads are still reading from the previous one.
template <typename T, typename Op> __device__ T block_reduce(const T value, const Op op)
{
  using BlockReduce = cub::BlockReduce<T, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result;
  __syncthreads();
  const T reduced = BlockReduce(temp_storage).Reduce(value, op);
  if (threadIdx.x == 0)
  {
    result = reduced;
  }
  __syncthreads();
  return result;
}

/// Launches `kernel` with `grid` blocks of `block` threads on `stream`, then checks the launch.
template <typename... Params, typename... Args>
void launch(void (*kernel)(Params...), unsigned int grid, unsigned int block, cudaStream_t stream, Args &&...args)
{
  kernel<<<grid, block, 0, stream>>>(std::forward<Args>(args)...);
  CHECK_GPU(cudaGetLastError());
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

namespace detail
{
/// Blocks of block_size threads of `kernel` that can be resident on the current device at once: the most a cooperative
/// launch may use. Computed on first use per kernel, then cached; assumes one device per process.
inline unsigned int max_cooperative_blocks(const void *const kernel)
{
  static std::mutex mutex;
  static std::unordered_map<const void *, unsigned int> cache;
  const std::scoped_lock lock{mutex};
  if (const auto found = cache.find(kernel); found != cache.end())
  {
    return found->second;
  }

  int device = 0;
  CHECK_GPU(cudaGetDevice(&device));
  int cooperative_launch = 0;
  CHECK_GPU(cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device));
  if (cooperative_launch == 0)
  {
    throw std::runtime_error("launch_cooperative: device does not support cooperative launch");
  }
  int multiprocessor_count = 0;
  CHECK_GPU(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_multiprocessor = 0;
  CHECK_GPU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor, kernel,
                                                          static_cast<int>(block_size), 0));
  const auto blocks = static_cast<unsigned int>(blocks_per_multiprocessor * multiprocessor_count);
  if (blocks == 0)
  {
    throw std::runtime_error("launch_cooperative: no resident blocks available");
  }
  return cache.emplace(kernel, blocks).first->second;
}
} // namespace detail

/// Launches `kernel` cooperatively on `stream`, so that it may call grid.sync(). One thread per element of [0, n), but
/// at most as many blocks as can be resident at once: the kernel must cover [0, n) with a grid-stride loop.
template <typename... Params, typename... Args>
void launch_cooperative(void (*kernel)(Params...), const std::size_t n, cudaStream_t stream, Args &&...args)
{
  assert(n > 0);
  const std::size_t blocks = std::min<std::size_t>(
      (n + block_size - 1) / block_size, detail::max_cooperative_blocks(reinterpret_cast<const void *>(kernel)));
  cudaLaunchAttribute cooperative{};
  cooperative.id = cudaLaunchAttributeCooperative;
  cooperative.val.cooperative = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned int>(blocks));
  config.blockDim = dim3(block_size);
  config.stream = stream;
  config.attrs = &cooperative;
  config.numAttrs = 1;
  CHECK_GPU(cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...));
}
} // namespace breifast::gpu
