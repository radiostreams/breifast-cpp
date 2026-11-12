#include <algorithm>
#include <cassert>
#include <cstdint>

#include "gpu_band.hpp"
#include "gpu_kernels.hpp"
#include "gpu_device_utils.cuh"
#include "radix_keys.cuh"

namespace breifast::gpu
{
namespace
{
/// Blocks of the histogram kernel at most: a few waves on any current GPU.
constexpr unsigned int histogram_max_blocks = 1024;

// One byte per pass, most significant first, of the keys whose resolved high bytes equal the prefix. A block counts
// into a histogram of its own in shared memory, then adds it to the device-wide 256-bin one once. The grid is capped
// and strides over the values, so that the merge costs one atomic per bin per block instead of one per value.
__global__ void radix_histogram_kernel(const float *values, size_t n, uint32_t *histogram, int shift, bool abs_mode,
                                       const uint32_t *state)
{
  __shared__ uint32_t local[radix_bins];
  for (size_t bin = threadIdx.x; bin < radix_bins; bin += blockDim.x)
  {
    local[bin] = 0;
  }
  __syncthreads();

  const uint32_t prefix = state[1];
  const uint32_t coverage = state[2];
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = global_index(); idx < n; idx += stride)
  {
    const uint32_t key = abs_mode ? abs_key(values[idx]) : signed_key(values[idx]);
    if ((key & coverage) == prefix)
    {
      atomicAdd(&local[(key >> shift) & 0xFFU], 1U);
    }
  }
  __syncthreads();

  for (size_t bin = threadIdx.x; bin < radix_bins; bin += blockDim.x)
  {
    if (local[bin] != 0)
    {
      atomicAdd(&histogram[bin], local[bin]);
    }
  }
}

#if 0
// Variant to revisit: one private histogram per thread, so no atomics and no bank conflicts. Disabled because it does
// not fit: radix_bins * block_size counters are 256 KB of shared memory, 48 KB at most (ptxas: "uses too much shared
// data"). Idea to explore: fewer bins per sweep.
__global__ void radix_histogram_private_kernel(const float *values, size_t n, uint32_t *histogram, int shift,
                                               bool abs_mode, const uint32_t *state)
{
  __shared__ uint32_t local[radix_bins * block_size];
  for (size_t bin = 0; bin < radix_bins; ++bin)
  {
    local[block_size * bin + threadIdx.x] = 0;
  }
  /*for (size_t bin = threadIdx.x; bin < radix_bins; bin += blockDim.x)
  {
    local[bin] = 0;
  }*/
  __syncthreads();

  const uint32_t prefix = state[1];
  const uint32_t coverage = state[2];
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = global_index(); idx < n; idx += stride)
  {
    const uint32_t key = abs_mode ? abs_key(values[idx]) : signed_key(values[idx]);
    if ((key & coverage) == prefix)
    {
      const auto bin = (key >> shift) & 0xFFU;
      ++local[bin * block_size + threadIdx.x];
    }
  }
  __syncthreads();

  for (size_t bin = threadIdx.x; bin < radix_bins; bin += blockDim.x)
  {
    for (size_t i = 1; i < block_size; ++i)
    {
      local[bin * block_size] += local[bin * block_size + i];
    }
    if (local[bin * block_size] != 0)
    {
      atomicAdd(&histogram[bin], local[bin * block_size]);
    }
  }
}
#endif

// Single thread. state[0..2] are below, prefix and coverage.
__global__ void radix_scan_kernel(const uint32_t *histogram, size_t rank, int shift, uint32_t *state)
{
  select_radix_digit(histogram, rank, shift, state[0], state[1], state[2]);
}

// Even n only: the largest key strictly below the resolved rank-k key, to resolve rank k-1 when it is not a repeat.
__global__ void radix_refine_kernel(const float *values, size_t n, bool abs_mode, const uint32_t *state,
                                    uint32_t *lower_max)
{
  const size_t idx = global_index();
  if (idx < n)
  {
    const uint32_t key = abs_mode ? abs_key(values[idx]) : signed_key(values[idx]);
    if (key < state[1])
    {
      atomicMax(lower_max, key);
    }
  }
}

// Single thread. state[1] is the full rank-n/2 key.
__global__ void radix_finalize_kernel(const uint32_t *state, const uint32_t *lower_max, bool abs_mode, size_t n,
                                      float scale, float *out_median)
{
  *out_median = median_of_keys(n, state[0], state[1], *lower_max, abs_mode) * scale;
}

void gpu_median_impl(const float *values, size_t n, float *out_median, bool abs_mode, float scale,
                     MedianScratch &scratch, cudaStream_t stream)
{
  static_assert(MedianScratch::histogram_bins == radix_bins);
  assert(n > 0);
  assert(scratch.words.size() == radix_bins + 4);

  uint32_t *histogram = scratch.words.data();
  uint32_t *state = histogram + radix_bins; // below, prefix, coverage
  uint32_t *lower_max = state + 3;
  // The scratch still holds the previous call's state; the histogram is cleared on every pass below.
  CHECK_GPU(cudaMemsetAsync(state, 0, 4 * sizeof(uint32_t), stream));

  const auto histogram_grid =
      static_cast<unsigned int>(std::min<size_t>((n + block_size - 1) / block_size, histogram_max_blocks));
  for (int shift = 24; shift >= 0; shift -= 8)
  {
    CHECK_GPU(cudaMemsetAsync(histogram, 0, radix_bins * sizeof(uint32_t), stream));
    launch(radix_histogram_kernel, histogram_grid, block_size, stream, values, n, histogram, shift, abs_mode, state);
    launch(radix_scan_kernel, 1, 1, stream, histogram, n / 2, shift, state);
  }

  if (n % 2 == 0)
  {
    launch_per_element(radix_refine_kernel, n, stream, values, n, abs_mode, state, lower_max);
  }

  launch(radix_finalize_kernel, 1, 1, stream, state, lower_max, abs_mode, n, scale, out_median);
}
} // namespace

void gpu_median(const float *values, size_t n, float *out_median, MedianScratch &scratch, cudaStream_t stream)
{
  gpu_median_impl(values, n, out_median, false, 1.F, scratch, stream);
}

void gpu_median_abs(const float *values, size_t n, float *out_median, float scale, MedianScratch &scratch,
                    cudaStream_t stream)
{
  gpu_median_impl(values, n, out_median, true, scale, scratch, stream);
}
} // namespace breifast::gpu
