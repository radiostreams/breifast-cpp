#include <cassert>
#include <limits>

#include "gpu_runtime.hpp"
#include "gpu_kernels.hpp"
#include "gpu_device_utils.cuh"

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

__global__ void weighted_band_image_kernel(const float *cube, const float *weights, const float *mean_image,
                                           float *out_plane, size_t n_pixels, long band_start, long band_stop,
                                           float inv_band_size)
{
  const auto i_pixel = global_index();
  if (i_pixel < n_pixels)
  {
    // Summed here rather than passed in, so that a captured CUDA graph replays for another plane's weights. Same
    // sequential order as the host std::accumulate, so the same float.
    float weights_sum = 0.F;
    for (long i_channel = band_start; i_channel < band_stop; ++i_channel)
    {
      weights_sum += weights[i_channel];
    }
    const float inv_weights_sum = 1.F / weights_sum;

    float result = 0.0f;
    for (long i_channel = band_start; i_channel < band_stop; ++i_channel)
    {
      const auto weight = weights[i_channel];
      const float value = cube[i_channel * n_pixels + i_pixel];
      result += weight * value * inv_weights_sum;
      if (mean_image != nullptr)
      {
        result -= mean_image[i_channel * n_pixels + i_pixel] * inv_band_size;
      }
    }
    out_plane[i_pixel] = result;
  }
}

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
                             size_t n_pixels, long band_start, long band_stop, float inv_band_size, cudaStream_t stream)
{
  assert(band_start >= 0 && band_start < band_stop);
  assert(n_pixels > 0);
  launch_per_element(weighted_band_image_kernel, n_pixels, stream, cube, weights, mean_image, out_plane, n_pixels,
                     band_start, band_stop, inv_band_size);
}

void gpu_spectral_variance_moments(const float *cube, const bool *nonzero_channels, float *out_sum2, float *out_sum3,
                                   float *out_sum4, size_t n_pixels, int n_channels, cudaStream_t stream)
{
  assert(n_pixels > 0);
  launch_per_element(spectral_variance_moments_kernel, n_pixels, stream, cube, nonzero_channels, out_sum2, out_sum3,
                     out_sum4, n_pixels, n_channels);
}

void gpu_spectral_variance_stats_host(const float *cube, const bool *nonzero_channels, size_t n_y, size_t n_x,
                                      size_t n_channels, GpuSpectralVarianceHostResult &result, cudaStream_t stream)
{
  assert(n_y > 0 && n_x > 0 && n_channels > 0);
  assert(cube != nullptr && nonzero_channels != nullptr);
  assert(n_y <= std::numeric_limits<size_t>::max() / n_x);
  assert(n_channels <= static_cast<size_t>(std::numeric_limits<int>::max()));

  const size_t n_pixels = n_y * n_x;

  // Per plane, not per band; the slot owns them in Phase 4.
  DeviceVector<float> sum_pow2(n_pixels);
  DeviceVector<float> sum_pow3(n_pixels);
  DeviceVector<float> sum_pow4(n_pixels);
  gpu_spectral_variance_moments(cube, nonzero_channels, sum_pow2, sum_pow3, sum_pow4, n_pixels,
                                static_cast<int>(n_channels), stream);

  result.sum_pow2.resize(n_pixels);
  result.sum_pow3.resize(n_pixels);
  result.sum_pow4.resize(n_pixels);
  result.nonzero_channels_mask.resize(n_channels);

  CHECK_GPU(cudaMemcpyAsync(result.sum_pow2.data(), sum_pow2.data(), n_pixels * sizeof(float), cudaMemcpyDeviceToHost,
                            stream));
  CHECK_GPU(cudaMemcpyAsync(result.sum_pow3.data(), sum_pow3.data(), n_pixels * sizeof(float), cudaMemcpyDeviceToHost,
                            stream));
  CHECK_GPU(cudaMemcpyAsync(result.sum_pow4.data(), sum_pow4.data(), n_pixels * sizeof(float), cudaMemcpyDeviceToHost,
                            stream));
  CHECK_GPU(cudaMemcpyAsync(result.nonzero_channels_mask.data(), nonzero_channels, n_channels * sizeof(bool),
                            cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaStreamSynchronize(stream));
}
} // namespace breifast::gpu
