#include <algorithm>
#include <cassert>
#include <cstdint>

#include <cub/block/block_scan.cuh>

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

// One pass of the radix select, most significant byte first, over the keys whose resolved high bytes equal the prefix.
// A block counts into a histogram of its own in shared memory, then adds it to the device-wide 256-bin one once; the
// grid is capped and strides over the values, so that the merge costs one atomic per bin per block. Scratch:
// histogram[256], then state = below, prefix, coverage, then lower_max and done (blocks finished with this pass). The
// last block to finish picks the byte for the next pass and, after the last pass, writes the median; it clears the
// histogram and the counter behind it.
__global__ void radix_select_pass_kernel(const float *values, size_t n, uint32_t *histogram, int shift, bool abs_mode,
                                         uint32_t *state, uint32_t *lower_max, uint32_t *done, float scale,
                                         float *out_median)
{
  static_assert(block_size == radix_bins, "the last block scans one bin per thread");
  __shared__ uint32_t local[radix_bins];
  local[threadIdx.x] = 0;
  __syncthreads();

  const uint32_t below = state[0];
  const uint32_t prefix = state[1];
  const uint32_t coverage = state[2];
  // Last pass, even n: also the largest key below the 24-bit prefix, for rank n/2 - 1.
  const bool track_lower = shift == 0 && n % 2 == 0;
  uint32_t local_lower = 0;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = global_index(); idx < n; idx += stride)
  {
    const uint32_t key = abs_mode ? abs_key(values[idx]) : signed_key(values[idx]);
    if ((key & coverage) == prefix)
    {
      atomicAdd(&local[(key >> shift) & 0xFFU], 1U);
    }
    else if (track_lower && key < prefix)
    {
      local_lower = max(local_lower, key);
    }
  }
  __syncthreads();
  if (local[threadIdx.x] != 0)
  {
    atomicAdd(&histogram[threadIdx.x], local[threadIdx.x]);
  }
  if (track_lower)
  {
    const uint32_t block_lower = block_reduce(local_lower, UintMax{});
    if (threadIdx.x == 0 && block_lower != 0)
    {
      atomicMax(lower_max, block_lower);
    }
  }

  __threadfence();
  __shared__ bool last_block;
  if (threadIdx.x == 0)
  {
    last_block = atomicAdd(done, 1U) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last_block)
  {
    return;
  }

  // Every block's counts are in: the bin that holds rank n/2.
  const uint32_t bin = threadIdx.x;
  const uint32_t count = __ldcg(&histogram[bin]);
  using BlockScan = cub::BlockScan<uint32_t, block_size>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ uint32_t selected_bin;
  uint32_t before = 0;
  BlockScan(scan_storage).ExclusiveSum(count, before);
  const size_t first = size_t{below} + before;
  const size_t rank = n / 2;
  if (count != 0 && first <= rank && rank < first + count)
  {
    selected_bin = bin;
    state[0] = static_cast<uint32_t>(first);
    state[1] = prefix | (bin << shift);
    state[2] = coverage | (0xFFU << shift);
  }
  __syncthreads();

  if (shift == 0)
  {
    // Rank n/2 - 1 when it is not a repeat: the highest non-empty bin below the selected one, else the largest key
    // below the 24-bit prefix.
    const uint32_t lower_bin = block_reduce(count != 0 && bin < selected_bin ? bin + 1 : 0U, UintMax{});
    if (threadIdx.x == 0)
    {
      const uint32_t key = prefix | selected_bin;
      const uint32_t lower = lower_bin != 0 ? (prefix | (lower_bin - 1)) : __ldcg(lower_max);
      *out_median = median_of_keys(n, __ldcg(&state[0]), key, lower, abs_mode) * scale;
    }
  }

  histogram[bin] = 0; // every thread has read its bin: the next pass starts from zero
  if (threadIdx.x == 0)
  {
    *done = 0;
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

void gpu_median_impl(const float *values, size_t n, float *out_median, bool abs_mode, float scale,
                     MedianScratch &scratch, cudaStream_t stream)
{
  static_assert(MedianScratch::histogram_bins == radix_bins);
  assert(n > 0);
  assert(scratch.words.size() == radix_bins + 5);

  uint32_t *histogram = scratch.words.data();
  uint32_t *state = histogram + radix_bins; // below, prefix, coverage
  uint32_t *lower_max = state + 3;
  uint32_t *done = state + 4;
  // One memset per median: the passes clear the histogram and the block counter behind them.
  CHECK_GPU(cudaMemsetAsync(scratch.words.data(), 0, scratch.words.size() * sizeof(uint32_t), stream));

  const auto histogram_grid =
      static_cast<unsigned int>(std::min<size_t>((n + block_size - 1) / block_size, histogram_max_blocks));
  for (int shift = 24; shift >= 0; shift -= 8)
  {
    launch(radix_select_pass_kernel, histogram_grid, block_size, stream, values, n, histogram, shift, abs_mode, state,
           lower_max, done, scale, out_median);
  }
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
