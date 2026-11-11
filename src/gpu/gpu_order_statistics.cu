#include <cassert>
#include <cstdint>

#include "gpu_band_workspace.cuh"
#include "gpu_kernels.hpp"
#include "gpu_device_utils.cuh"
#include "radix_keys.cuh"

namespace breifast::gpu
{
namespace
{
// One byte per pass, most significant first, of the keys whose resolved high bytes equal the prefix. Counts into a
// device-wide 256-bin histogram.
__global__ void radix_histogram_kernel(const float *values, size_t n, uint32_t *histogram, int shift, bool abs_mode,
                                       const uint32_t *state)
{
  const size_t idx = global_index();
  if (idx < n)
  {
    const uint32_t key = abs_mode ? abs_key(values[idx]) : signed_key(values[idx]);
    if ((key & state[2]) == state[1])
    {
      atomicAdd(&histogram[(key >> shift) & 0xFFU], 1U);
    }
  }
}

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
                     MedianScratch &scratch, GpuStream stream)
{
  static_assert(MedianScratch::histogram_bins == radix_bins);
  assert(n > 0);
  assert(scratch.words.size() == radix_bins + 4);

  uint32_t *histogram = scratch.words.data();
  uint32_t *state = histogram + radix_bins; // below, prefix, coverage
  uint32_t *lower_max = state + 3;
  // The scratch still holds the previous call's state; the histogram is cleared on every pass below.
  CHECK_GPU(cudaMemsetAsync(state, 0, 4 * sizeof(uint32_t), stream));

  for (int shift = 24; shift >= 0; shift -= 8)
  {
    CHECK_GPU(cudaMemsetAsync(histogram, 0, radix_bins * sizeof(uint32_t), stream));
    launch_per_element(radix_histogram_kernel, n, stream, values, n, histogram, shift, abs_mode, state);
    launch(radix_scan_kernel, 1, 1, stream, histogram, n / 2, shift, state);
  }

  if (n % 2 == 0)
  {
    launch_per_element(radix_refine_kernel, n, stream, values, n, abs_mode, state, lower_max);
  }

  launch(radix_finalize_kernel, 1, 1, stream, state, lower_max, abs_mode, n, scale, out_median);
}
} // namespace

void gpu_median(const float *values, size_t n, float *out_median, MedianScratch &scratch, GpuStream stream)
{
  gpu_median_impl(values, n, out_median, false, 1.F, scratch, stream);
}

void gpu_median_abs(const float *values, size_t n, float *out_median, float scale, MedianScratch &scratch,
                    GpuStream stream)
{
  gpu_median_impl(values, n, out_median, true, scale, scratch, stream);
}
} // namespace breifast::gpu
