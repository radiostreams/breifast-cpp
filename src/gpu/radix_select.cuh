#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>

#include <cub/block/block_scan.cuh>

#include "gpu_device_utils.cuh"
#include "gpu_kernels.hpp"
#include "radix_keys.cuh"

namespace breifast::gpu
{
/// Blocks of a radix select pass at most: a few waves on any current GPU.
inline constexpr unsigned int select_max_blocks = 1024;

/// The median's radix select: three digits of 11, 11 and 10 bits, most significant first.
inline constexpr uint32_t select_bins = 2048;
inline constexpr int select_shifts[] = {21, 10, 0};
inline constexpr uint32_t select_masks[] = {0x7FFU, 0x7FFU, 0x3FFU};
inline constexpr uint32_t bins_per_thread = select_bins / block_size; // the last block scans 8 bins per thread
static_assert(MedianScratch::histogram_bins == select_bins);

/// The values a radix select pass counts: `load4(i4)` gives values 4 * i4 to 4 * i4 + 3, `operator()(i)` value i. This
/// one reads them from memory. A kernel that produces the values folds the first pass in through a source that
/// computes them.
struct RadixValues
{
  const float *values;
  __device__ float4 load4(const size_t i4) const { return reinterpret_cast<const float4 *>(values)[i4]; }
  __device__ float operator()(const size_t i) const { return values[i]; }
};

// One pass of the radix select over the keys whose resolved high digits equal the prefix. A block counts into a
// histogram of its own in shared memory, then adds it to the device-wide one once; the grid is capped and strides over
// the values, so that the merge costs one atomic per bin per block. Scratch: histogram[select_bins], then state =
// below, prefix, coverage, then lower_max and done (blocks finished with this pass). The last block to finish picks the
// digit for the next pass and, after the last pass, writes the median; it clears the histogram and the counter behind
// it.
template <typename Source>
__global__ void radix_select_pass_kernel(const Source source, size_t n, uint32_t *histogram, int shift, uint32_t mask,
                                         bool abs_mode, uint32_t *state, uint32_t *lower_max, uint32_t *done,
                                         float scale, float *out_median)
{
  static_assert(select_bins % block_size == 0);
  __shared__ uint32_t local[select_bins];
  for (uint32_t bin = threadIdx.x; bin < select_bins; bin += blockDim.x)
  {
    local[bin] = 0;
  }
  __syncthreads();

  const uint32_t below = state[0];
  const uint32_t prefix = state[1];
  const uint32_t coverage = state[2];
  // Last pass, even n: also the largest key below the 22-bit prefix, for rank n/2 - 1.
  const bool track_lower = shift == 0 && n % 2 == 0;
  uint32_t local_lower = 0;
  // Four values per load and two loads in flight per iteration: one 4-byte load per thread does not cover the memory
  // latency (the pass stalled on its loads, not on the shared atomics).
  const auto count_value = [&](const float value)
  {
    const uint32_t key = abs_mode ? abs_key(value) : signed_key(value);
    if ((key & coverage) == prefix)
    {
      atomicAdd(&local[(key >> shift) & mask], 1U);
    }
    else if (track_lower && key < prefix)
    {
      local_lower = max(local_lower, key);
    }
  };
  const auto count_values = [&](const float4 v)
  {
    count_value(v.x);
    count_value(v.y);
    count_value(v.z);
    count_value(v.w);
  };

  const size_t n4 = n / 4;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t i = global_index();
  for (; i + stride < n4; i += 2 * stride)
  {
    const float4 a = source.load4(i);
    const float4 b = source.load4(i + stride);
    count_values(a);
    count_values(b);
  }
  if (i < n4)
  {
    count_values(source.load4(i));
  }
  for (size_t idx = 4 * n4 + global_index(); idx < n; idx += stride) // the last n % 4 values
  {
    count_value(source(idx));
  }
  __syncthreads();
  for (uint32_t bin = threadIdx.x; bin < select_bins; bin += blockDim.x)
  {
    if (local[bin] != 0)
    {
      atomicAdd(&histogram[bin], local[bin]);
    }
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

  // Every block's counts are in: the bin that holds rank n/2. Thread t owns bins [8t, 8t + 8).
  uint32_t counts[bins_per_thread];
  uint32_t thread_total = 0;
  for (uint32_t j = 0; j < bins_per_thread; ++j)
  {
    counts[j] = __ldcg(&histogram[threadIdx.x * bins_per_thread + j]);
    thread_total += counts[j];
  }
  using BlockScan = cub::BlockScan<uint32_t, block_size>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ uint32_t selected_bin;
  uint32_t before = 0;
  BlockScan(scan_storage).ExclusiveSum(thread_total, before);
  const size_t rank = n / 2;
  size_t first = size_t{below} + before;
  for (uint32_t j = 0; j < bins_per_thread; ++j)
  {
    if (counts[j] != 0 && first <= rank && rank < first + counts[j])
    {
      const uint32_t bin = threadIdx.x * bins_per_thread + j;
      selected_bin = bin;
      state[0] = static_cast<uint32_t>(first);
      state[1] = prefix | (bin << shift);
      state[2] = coverage | (mask << shift);
    }
    first += counts[j];
  }
  __syncthreads();

  if (shift == 0)
  {
    // Rank n/2 - 1 when it is not a repeat: the highest non-empty bin below the selected one, else the largest key
    // below the 22-bit prefix.
    uint32_t thread_lower = 0;
    for (uint32_t j = 0; j < bins_per_thread; ++j)
    {
      const uint32_t bin = threadIdx.x * bins_per_thread + j;
      if (counts[j] != 0 && bin < selected_bin)
      {
        thread_lower = bin + 1;
      }
    }
    const uint32_t lower_bin = block_reduce(thread_lower, UintMax{});
    if (threadIdx.x == 0)
    {
      const uint32_t key = prefix | selected_bin;
      const uint32_t lower = lower_bin != 0 ? (prefix | (lower_bin - 1)) : __ldcg(lower_max);
      *out_median = median_of_keys(n, __ldcg(&state[0]), key, lower, abs_mode) * scale;
    }
  }

  for (uint32_t j = 0; j < bins_per_thread; ++j) // every thread has read its bins: the next pass starts from zero
  {
    histogram[threadIdx.x * bins_per_thread + j] = 0;
  }
  if (threadIdx.x == 0)
  {
    *done = 0;
  }
}

/// Median of the `n` values of `values` (of their absolute values in `abs_mode`), times `scale`, into `out_median`: three
/// radix select passes on `stream`. The first pass reads the values through `first`, so that the kernel producing them
/// can be the first pass; the other two read `values`, where `first` must have left them.
template <typename First>
void radix_median(const First &first, const float *values, size_t n, bool abs_mode, float scale, float *out_median,
                  MedianScratch &scratch, cudaStream_t stream)
{
  assert(n > 0);
  assert(scratch.words.size() == select_bins + 5);
  assert(reinterpret_cast<std::uintptr_t>(values) % alignof(float4) == 0); // whole buffers: 256-byte aligned

  uint32_t *histogram = scratch.words.data();
  uint32_t *state = histogram + select_bins; // below, prefix, coverage
  uint32_t *lower_max = state + 3;
  uint32_t *done = state + 4;
  // One memset per median: the passes clear the histogram and the block counter behind them.
  CHECK_GPU(cudaMemsetAsync(scratch.words.data(), 0, scratch.words.size() * sizeof(uint32_t), stream));

  const auto grid = static_cast<unsigned int>(std::min<size_t>((n + block_size - 1) / block_size, select_max_blocks));
  launch(radix_select_pass_kernel<First>, grid, block_size, stream, first, n, histogram, select_shifts[0],
         select_masks[0], abs_mode, state, lower_max, done, scale, out_median);
  for (int pass = 1; pass < 3; ++pass)
  {
    launch(radix_select_pass_kernel<RadixValues>, grid, block_size, stream, RadixValues{values}, n, histogram,
           select_shifts[pass], select_masks[pass], abs_mode, state, lower_max, done, scale, out_median);
  }
}
} // namespace breifast::gpu
