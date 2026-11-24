#include <cassert>

#include "gpu_runtime.hpp"
#include "gpu_kernels.hpp"
#include "gpu_device_utils.cuh"
#include "radix_select.cuh"

namespace breifast::gpu
{
namespace
{
__global__ void mask_or_kernel(const bool *accept_mask, size_t n_pixels, long band_start, long band_stop, bool *out_mask,
                               int *out_any)
{
  const auto i_pixel = global_index();
  bool accept = false;
  if (i_pixel < n_pixels)
  {
    for (long i_channel = band_start; i_channel < band_stop; ++i_channel)
    {
      accept |= accept_mask[i_channel * n_pixels + i_pixel];
    }
    out_mask[i_pixel] = accept;
  }
  const auto block_any = __syncthreads_or(accept);
  if (threadIdx.x == 0 && block_any)
  {
    atomicOr(out_any, 1);
  }
}

// The weighted band image, stored as the first pass of its median reads it.
struct WeightedBand
{
  const float *cube;
  const float *weights;
  const float *mean_image;
  float *out_plane;
  size_t n_pixels;
  long band_start;
  long band_stop;
  float inv_band_size;

  /// At least 4 blocks per SM (64 registers) for its radix pass: left free, it took 98 and 2 blocks, too few warps to
  /// cover its loads.
  static constexpr unsigned int min_blocks_per_sm = 4;

  // Summed here rather than passed in, so that a captured CUDA graph replays for another plane's weights. Same
  // sequential order as the host std::accumulate, so the same float.
  __device__ float inv_weights_sum() const
  {
    float weights_sum = 0.F;
    for (long i_channel = band_start; i_channel < band_stop; ++i_channel)
    {
      weights_sum += weights[i_channel];
    }
    return 1.F / weights_sum;
  }
  __device__ float operator()(const size_t i_pixel) const
  {
    const float inv_sum = inv_weights_sum();
    float result = 0.F;
    for (long i_channel = band_start; i_channel < band_stop; ++i_channel)
    {
      result += weights[i_channel] * cube[i_channel * n_pixels + i_pixel] * inv_sum;
      if (mean_image != nullptr)
      {
        result -= mean_image[i_channel * n_pixels + i_pixel] * inv_band_size;
      }
    }
    out_plane[i_pixel] = result;
    return result;
  }
  __device__ float4 load4(const size_t i4) const
  {
    if (n_pixels % 4 != 0) // the channel slices are not 16-byte aligned
    {
      return {(*this)(4 * i4), (*this)(4 * i4 + 1), (*this)(4 * i4 + 2), (*this)(4 * i4 + 3)};
    }
    const float inv_sum = inv_weights_sum();
    float4 result = {0.F, 0.F, 0.F, 0.F};
    for (long i_channel = band_start; i_channel < band_stop; ++i_channel)
    {
      const float weight = weights[i_channel];
      const float4 value = reinterpret_cast<const float4 *>(cube + i_channel * n_pixels)[i4];
      result.x += weight * value.x * inv_sum;
      result.y += weight * value.y * inv_sum;
      result.z += weight * value.z * inv_sum;
      result.w += weight * value.w * inv_sum;
      if (mean_image != nullptr)
      {
        const float4 mean = reinterpret_cast<const float4 *>(mean_image + i_channel * n_pixels)[i4];
        result.x -= mean.x * inv_band_size;
        result.y -= mean.y * inv_band_size;
        result.z -= mean.z * inv_band_size;
        result.w -= mean.w * inv_band_size;
      }
    }
    reinterpret_cast<float4 *>(out_plane)[i4] = result;
    return result;
  }
};

__global__ void spectral_variance_moments_kernel(const float *cube, const bool *nonzero_channels, float *out_sum2,
                                                 float *out_sum3, float *out_sum4, size_t n_pixels, size_t n_channels)
{
  const auto i_pixel = global_index();
  if (i_pixel < n_pixels)
  {
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    for (size_t i_channel = 0; i_channel < n_channels; ++i_channel)
    {
      if (nonzero_channels[i_channel])
      {
        const float value = cube[i_channel * n_pixels + i_pixel];
        const auto value2 = value * value;
        const auto value3 = value2 * value;
        const auto value4 = value3 * value;
        sum2 += value2;
        sum3 += value3;
        sum4 += value4;
      }
    }
    out_sum2[i_pixel] = sum2;
    out_sum3[i_pixel] = sum3;
    out_sum4[i_pixel] = sum4;
  }
}
} // namespace

void gpu_mask_or(const bool *accept_mask, size_t n_pixels, long band_start, long band_stop, bool *out_mask,
                 int *out_any, cudaStream_t stream)
{
  assert(band_start >= 0 && band_start < band_stop);
  assert(n_pixels > 0);

  CHECK_GPU(cudaMemsetAsync(out_any, 0, sizeof(int), stream));
  launch_per_element(mask_or_kernel, n_pixels, stream, accept_mask, n_pixels, band_start, band_stop, out_mask,
                     out_any);
}

void gpu_weighted_band_image(const float *cube, const float *weights, const float *mean_image, float *out_plane,
                             size_t n_pixels, long band_start, long band_stop, float inv_band_size,
                             float *out_median_abs, float scale, MedianScratch &scratch, cudaStream_t stream)
{
  assert(band_start >= 0 && band_start < band_stop);
  assert(n_pixels > 0);
  radix_median(WeightedBand{cube, weights, mean_image, out_plane, n_pixels, band_start, band_stop, inv_band_size},
               out_plane, n_pixels, true, scale, out_median_abs, scratch, stream);
}

void gpu_spectral_variance_moments(const float *cube, const bool *nonzero_channels, float *out_sum2, float *out_sum3,
                                   float *out_sum4, size_t n_pixels, int n_channels, cudaStream_t stream)
{
  assert(n_pixels > 0);
  launch_per_element(spectral_variance_moments_kernel, n_pixels, stream, cube, nonzero_channels, out_sum2, out_sum3,
                     out_sum4, n_pixels, n_channels);
}
} // namespace breifast::gpu
