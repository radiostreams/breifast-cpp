#include <cassert>
#include <cfloat>

#include <cuda/std/functional>

#include "gpu_kernels.hpp"

#include "gpu_device_utils.cuh"
#include "radix_keys.cuh"

namespace breifast::gpu
{
namespace
{
__global__ void donut_median_abs_kernel(const StampWindows windows, float scale, float *out_lmad)
{
  const auto i_stamp = blockIdx.x;
  const Stamp stamp(windows, i_stamp);
  const size_t stamp_pixels = stamp.size();

  __shared__ uint32_t histogram[radix_bins];
  __shared__ uint32_t non_zero_count;
  __shared__ uint32_t below;
  __shared__ uint32_t prefix;
  __shared__ uint32_t coverage;
  __shared__ uint32_t lower_max;
  if (threadIdx.x == 0)
  {
    below = 0;
    prefix = 0;
    coverage = 0;
    lower_max = 0;
  }

  // One byte of the nonzero pixels' abs keys per pass. coverage starts at 0, so the first pass sees every nonzero pixel
  // and its histogram sums to their count.
  for (int shift = 24; shift >= 0; shift -= 8)
  {
    for (size_t i = threadIdx.x; i < radix_bins; i += blockDim.x)
    {
      histogram[i] = 0;
    }
    __syncthreads();

    for (size_t i_pixel = threadIdx.x; i_pixel < stamp_pixels; i_pixel += blockDim.x)
    {
      const float value = stamp[i_pixel];
      const uint32_t key = abs_key(value);
      if (value != 0.0f && (key & coverage) == prefix)
      {
        atomicAdd(&histogram[(key >> shift) & 0xFFU], 1U);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
      if (shift == 24)
      {
        non_zero_count = 0;
        for (uint32_t i = 0; i < radix_bins; ++i)
        {
          non_zero_count += histogram[i];
        }
      }
      if (non_zero_count > 0)
      {
        select_radix_digit(histogram, non_zero_count / 2, shift, below, prefix, coverage);
      }
    }
    __syncthreads();

    if (non_zero_count == 0)
    {
      if (threadIdx.x == 0)
      {
        out_lmad[i_stamp] = 0;
      }
      return;
    }
  }

  if (non_zero_count % 2 == 0)
  {
    for (size_t i_pixel = threadIdx.x; i_pixel < stamp_pixels; i_pixel += blockDim.x)
    {
      const float value = stamp[i_pixel];
      const uint32_t key = abs_key(value);
      if (value != 0.0f && key < prefix)
      {
        atomicMax(&lower_max, key);
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    out_lmad[i_stamp] = scale * median_of_keys(non_zero_count, below, prefix, lower_max, true);
  }
}

__global__ void stamp_abs_extreme_kernel(const StampWindows windows, float *out_extreme)
{
  const size_t i_stamp = blockIdx.x;
  const Stamp stamp(windows, blockIdx.x);
  const size_t stamp_pixels = stamp.size();

  float min_value = FLT_MAX;
  float max_value = -FLT_MAX;
  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    const float value = stamp[i];
    min_value = fminf(min_value, value);
    max_value = fmaxf(max_value, value);
  }

  min_value = block_reduce(min_value, FloatMin{});
  max_value = block_reduce(max_value, FloatMax{});

  if (threadIdx.x == 0)
  {
    out_extreme[i_stamp] = fmaxf(fabsf(min_value), fabsf(max_value));
  }
}

__global__ void gather_at_peaks_kernel(const float *plane, const float *noise_map, const float *rms_map,
                                       const float *selfnoise_raw, size_t n_x, const long *peak_y, const long *peak_x,
                                       unsigned int n_islands, float *flux, float *noise_at_peak, float *rms_at_peak,
                                       float *selfnoise_at_peak)
{
  const auto island = global_index();
  if (island >= n_islands)
  {
    return;
  }
  const auto pixel = static_cast<size_t>(peak_y[island]) * n_x + static_cast<size_t>(peak_x[island]);
  flux[island] = plane[pixel];
  noise_at_peak[island] = noise_map[pixel];
  rms_at_peak[island] = rms_map != nullptr ? rms_map[pixel] : 0.F;
  selfnoise_at_peak[island] = selfnoise_raw[pixel];
}

/// Sum, sum of squares and count of the nonzero pixels of a stamp.
struct Moments
{
  double sum;
  double sumsq;
  uint32_t count;
};

__host__ __device__ Moments operator+(const Moments &a, const Moments &b)
{
  return {a.sum + b.sum, a.sumsq + b.sumsq, a.count + b.count};
}

__global__ void donut_std_kernel(const StampWindows windows, float *out_std)
{
  const Stamp stamp(windows, blockIdx.x);
  const size_t stamps_pixels = stamp.size();

  Moments moments{0.0, 0.0, 0U};
  for (size_t i_pixel = threadIdx.x; i_pixel < stamps_pixels; i_pixel += blockDim.x)
  {
    const auto value = stamp[i_pixel];
    if (value != 0.F)
    {
      moments.sum += value;
      moments.sumsq += value * value;
      ++moments.count;
    }
  }

  const Moments total = block_reduce(moments, cuda::std::plus<>{});

  if (threadIdx.x == 0)
  {
    if (total.count == 0)
    {
      out_std[blockIdx.x] = 0.F;
      return;
    }
    const auto mean = total.sum / static_cast<float>(total.count);
    const auto meansq = total.sumsq / static_cast<float>(total.count);
    const auto var = meansq - mean * mean;
    out_std[blockIdx.x] = sqrtf(static_cast<float>(var));
  }
}

__global__ void stamp_beam_chi2_kernel(const StampWindows windows, const float *kernel, const float kernel_sum,
                                       const float *stamp_extreme, float inv_max_to_std_ratio, float *out_chi2)
{
  const auto i_stamp = blockIdx.x;
  const float adjacent_noise = stamp_extreme[i_stamp] * inv_max_to_std_ratio;

  const Stamp stamp(windows, i_stamp);
  const size_t stamp_pixels = stamp.size();

  float stamp_sum = 0.F;
  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    stamp_sum += stamp[i];
  }
  stamp_sum = block_reduce(stamp_sum, cuda::std::plus<>{});
  const auto scale = stamp_sum / kernel_sum;

  float resid2 = 0.F;
  uint32_t count = 0U;
  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    const auto value = stamp[i];
    if (value != 0.F)
    {
      const auto delta = value - scale * kernel[i];
      resid2 += delta * delta;
      ++count;
    }
  }
  resid2 = block_reduce(resid2, cuda::std::plus<>{});
  count = block_reduce(count, cuda::std::plus<>{});
  if (threadIdx.x == 0)
  {
    out_chi2[i_stamp] = (count > 0 && adjacent_noise != 0)
                            ? resid2 / (adjacent_noise * adjacent_noise * count)
                            : 0.F;
  }
}
} // namespace

void gpu_donut_median_abs(const StampWindows &stamps, unsigned int n_islands, float scale, float *out_lmad,
                          cudaStream_t stream)
{
  launch_per_island(donut_median_abs_kernel, n_islands, stream, stamps, scale, out_lmad);
}

void gpu_donut_std(const StampWindows &stamps, unsigned int n_islands, float *out_std, cudaStream_t stream)
{
  launch_per_island(donut_std_kernel, n_islands, stream, stamps, out_std);
}

void gpu_stamp_abs_extreme(const StampWindows &stamps, unsigned int n_islands, float *out_extreme, cudaStream_t stream)
{
  launch_per_island(stamp_abs_extreme_kernel, n_islands, stream, stamps, out_extreme);
}

void gpu_stamp_beam_chi2(const StampWindows &stamps, const float *kernel, const float kernel_sum,
                         unsigned int n_islands, const float *stamp_extreme, float inv_max_to_std_ratio,
                         float *out_chi2, cudaStream_t stream)
{
  launch_per_island(stamp_beam_chi2_kernel, n_islands, stream, stamps, kernel, kernel_sum, stamp_extreme,
                    inv_max_to_std_ratio, out_chi2);
}

void gpu_gather_at_peaks(const float *plane, const float *noise_map, const float *rms_map, const float *selfnoise_raw,
                         size_t n_x, const long *peak_y, const long *peak_x, unsigned int n_islands, float *flux,
                         float *noise_at_peak, float *rms_at_peak, float *selfnoise_at_peak, cudaStream_t stream)
{
  assert(n_islands > 0);
  launch_per_element(gather_at_peaks_kernel, n_islands, stream, plane, noise_map, rms_map, selfnoise_raw, n_x, peak_y,
                     peak_x, n_islands, flux, noise_at_peak, rms_at_peak, selfnoise_at_peak);
}
} // namespace breifast::gpu
