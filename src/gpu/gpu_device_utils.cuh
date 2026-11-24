#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#include "gpu_kernels.hpp"
#include "gpu_runtime.hpp"

namespace breifast::gpu
{
/// Threads per block of every kernel launched through the helpers below; block_reduce is specialised for it.
inline constexpr unsigned int block_size = 256;

/// Index of the calling thread in a one-dimensional grid, computed in size_t so that it cannot overflow.
__device__ inline std::size_t global_index()
{
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

/// Column of the calling thread in a two-dimensional grid. 32-bit: one side of a 2D grid stays far below 2^32 threads.
__device__ inline unsigned int global_index_x()
{
  return blockIdx.x * blockDim.x + threadIdx.x;
}

/// Row of the calling thread in a two-dimensional grid. 32-bit, like global_index_x.
__device__ inline unsigned int global_index_y()
{
  return blockIdx.y * blockDim.y + threadIdx.y;
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
struct UintMax
{
  __device__ uint32_t operator()(const uint32_t a, const uint32_t b) const { return a > b ? a : b; }
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

/// One island's stamp, read in place from the plane as StampWindows describes it: cell i of the (2 * radius + 1)^2
/// window, row major, is the plane value, or 0 outside the plane or inside the inner box.
class Stamp
{
public:
  __device__ Stamp(const StampWindows &windows, const unsigned int island)
      : plane_{windows.plane}, n_y_{static_cast<long>(windows.n_y)}, n_x_{static_cast<long>(windows.n_x)},
        first_y_{windows.peak_y[island] - windows.radius}, first_x_{windows.peak_x[island] - windows.radius},
        radius_{windows.radius}, side_{2 * windows.radius + 1},
        half_{windows.inner_half < windows.radius ? windows.inner_half : windows.radius}
  {
  }

  __device__ std::size_t size() const
  {
    return static_cast<std::size_t>(side_ * side_);
  }

  __device__ float operator[](const std::size_t i) const
  {
    const long row = static_cast<long>(i / static_cast<std::size_t>(side_));
    const long column = static_cast<long>(i % static_cast<std::size_t>(side_));
    const long y = first_y_ + row;
    const long x = first_x_ + column;
    const bool inside = y >= 0 && y < n_y_ && x >= 0 && x < n_x_;
    const long dr = row - radius_;
    const long dc = column - radius_;
    // half_ < 0 (no inner box) makes this false for every cell.
    const bool inner = dr >= -half_ && dr <= half_ && dc >= -half_ && dc <= half_;
    return inside && !inner ? plane_[y * n_x_ + x] : 0.F;
  }

private:
  const float *plane_;
  long n_y_;
  long n_x_;
  long first_y_;
  long first_x_;
  long radius_;
  long side_;
  long half_;
};
} // namespace breifast::gpu
