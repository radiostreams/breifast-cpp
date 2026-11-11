#include <cstdint>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "gpu_pipeline.hpp"
#include "gpu_device_utils.cuh"
#include "radix_keys.cuh"

namespace breifast::gpu
{
namespace
{
constexpr size_t n_digits = 4;
constexpr size_t histogram_bins = 1 << 8;

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

// Cumulative count over the bins, stopping where rank `k` lands. state[0] is the count below the current prefix, which
// this advances; state[1] and state[2] extend the prefix and the coverage with the chosen byte.
__global__ void radix_scan_kernel(const uint32_t *histogram, size_t rank, int shift, uint32_t *state)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    size_t cumulative = state[0];
    uint32_t bin = 0;
    for (; bin < histogram_bins; ++bin)
    {
      if (cumulative + histogram[bin] > rank)
      {
        break;
      }
      cumulative += histogram[bin];
    }
    state[0] = static_cast<uint32_t>(cumulative);
    state[1] |= bin << shift;
    state[2] |= 0xFFU << shift;
  }
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

// state[1] is the full rank-k key. Even n averages it with rank k-1: the same key when the rank-k key is not the first
// of its repeats, else the largest key below it.
__global__ void radix_finalize_kernel(const uint32_t *state, const uint32_t *lower_max, bool abs_mode, size_t n,
                                      float scale, float *out_median)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    const uint32_t hi_key = state[1];
    const float hi = abs_mode ? abs_value(hi_key) : signed_value(hi_key);
    if (n % 2 == 1)
    {
      *out_median = hi * scale;
      return;
    }
    const uint32_t lo_key = static_cast<size_t>(n / 2) > state[0] ? hi_key : lower_max[0];
    const float lo = abs_mode ? abs_value(lo_key) : signed_value(lo_key);
    *out_median = ((lo + hi) / 2.F) * scale;
  }
}

__global__ void write_nan_kernel(float *out)
{
  *out = __int_as_float(0x7FC00000);
}

void gpu_median_impl(const float *values, size_t n, float *out_median, bool abs_mode, float scale, GpuStream stream)
{
  if (n == 0)
  {
    launch(write_nan_kernel, 1, 1, stream, out_median);
    return;
  }

  constexpr size_t scratch_words = histogram_bins + 4; // 256 bins, then state[3] and lower_max[1]
  thrust::device_vector<uint32_t> scratch(scratch_words, 0);
  uint32_t *scratch_ptr = thrust::raw_pointer_cast(scratch.data());

  uint32_t *histogram = scratch_ptr;
  uint32_t *state = scratch_ptr + histogram_bins;      // below, prefix, coverage
  uint32_t *lower_max = scratch_ptr + histogram_bins + 3;

  const int shifts[n_digits] = {24, 16, 8, 0};
  for (int shift : shifts)
  {
    gpuMemsetAsync(histogram, 0, histogram_bins * sizeof(uint32_t), stream);
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

void gpu_median(const float *values, size_t n, float *out_median, GpuStream stream)
{
  gpu_median_impl(values, n, out_median, false, 1.F, stream);
}

void gpu_median_abs(const float *values, size_t n, float *out_median, float scale, GpuStream stream)
{
  gpu_median_impl(values, n, out_median, true, scale, stream);
}
} // namespace breifast::gpu
