#include <limits>
#include <stdexcept>

#include "gpu_pipeline.hpp"
#include "gpu_runtime.hpp"

namespace breifast::gpu
{
namespace
{
__global__ void mask_or_kernel(const bool *accept_mask, size_t n_pixels, long band_start, long band_stop, bool *out_mask,
                               int *out_any)
{
  const auto i_pixel = blockIdx.x * blockDim.x + threadIdx.x;
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
                                           float inv_weights_sum, float inv_band_size)
{
  const auto i_pixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_pixel < n_pixels)
  {
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
  const auto i_pixel = blockIdx.x * blockDim.x + threadIdx.x;
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

void gpu_mask_or(const bool *accept_mask, size_t n_pixels, long band_start, long band_stop, bool *out_mask, int *out_any,
                 GpuStream stream)
{
  if (band_start < 0 || band_start >= band_stop)
  {
    throw std::runtime_error("gpu_mask_or: invalid band");
  }
  if (n_pixels == 0)
  {
    throw std::runtime_error("gpu_mask_or: zero pixels");
  }

  gpuMemsetAsync(out_any, 0, sizeof(int), stream);

  constexpr size_t threads = 256;
  const dim3 block_size = {threads};
  const dim3 grid_size = {static_cast<unsigned int>((n_pixels + threads - 1) / threads)};

  mask_or_kernel<<<grid_size, block_size, 0, stream>>>(accept_mask, n_pixels, band_start, band_stop, out_mask, out_any);
  CHECK_LAST_GPU_ERROR();
}

void gpu_weighted_band_image(const float *cube, const float *weights, const float *mean_image, float *out_plane,
                             size_t n_pixels, long band_start, long band_stop, float inv_weights_sum, float inv_band_size,
                             GpuStream stream)
{
  if (band_start < 0 || band_start >= band_stop)
  {
    throw std::runtime_error("gpu_weighted_band_image: invalid band");
  }
  if (n_pixels == 0)
  {
    throw std::runtime_error("gpu_weighted_band_image: zero pixels");
  }

  constexpr size_t threads = 256;
  const dim3 block_size = {threads};
  const dim3 grid_size = {static_cast<unsigned int>((n_pixels + threads - 1) / threads)};

  weighted_band_image_kernel<<<grid_size, block_size, 0, stream>>>(
      cube, weights, mean_image, out_plane, n_pixels, band_start, band_stop, inv_weights_sum, inv_band_size);
  CHECK_LAST_GPU_ERROR();
}

void gpu_spectral_variance_moments(const float *cube, const bool *nonzero_channels, float *out_sum2, float *out_sum3,
                                   float *out_sum4, size_t n_pixels, int n_channels, GpuStream stream)
{
  if (n_pixels == 0)
  {
    throw std::runtime_error("gpu_spectral_variance_moments: zero pixels");
  }

  constexpr size_t threads = 256;
  const dim3 block_size = {threads};
  const dim3 grid_size = {static_cast<unsigned int>((n_pixels + threads - 1) / threads)};

  spectral_variance_moments_kernel<<<grid_size, block_size, 0, stream>>>(cube, nonzero_channels, out_sum2, out_sum3,
                                                                         out_sum4, n_pixels, n_channels);
  CHECK_LAST_GPU_ERROR();
}

void gpu_spectral_variance_stats_host(const float *cube, const bool *nonzero_channels, size_t n_y, size_t n_x,
                                      size_t n_channels, GpuSpectralVarianceHostResult &result, GpuStream stream)
{
  if (n_y == 0 || n_x == 0 || n_channels == 0)
  {
    throw std::runtime_error("gpu_spectral_variance_stats_host: zero dimension");
  }
  if (cube == nullptr || nonzero_channels == nullptr)
  {
    throw std::runtime_error("gpu_spectral_variance_stats_host: null input");
  }
  if (n_y > std::numeric_limits<size_t>::max() / n_x)
  {
    throw std::runtime_error("gpu_spectral_variance_stats_host: pixel count overflow");
  }
  if (n_channels > static_cast<size_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error("gpu_spectral_variance_stats_host: channel count exceeds int");
  }

  const size_t n_pixels = n_y * n_x;
  float *sum_pow2 = nullptr;
  float *sum_pow3 = nullptr;
  float *sum_pow4 = nullptr;

  try
  {
    gpuMallocAsync(&sum_pow2, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&sum_pow3, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&sum_pow4, n_pixels * sizeof(float), stream);
    gpu_spectral_variance_moments(cube, nonzero_channels, sum_pow2, sum_pow3, sum_pow4, n_pixels,
                                  static_cast<int>(n_channels), stream);

    result.sum_pow2.resize(n_pixels);
    result.sum_pow3.resize(n_pixels);
    result.sum_pow4.resize(n_pixels);
    result.nonzero_channels_mask.resize(n_channels);

    gpuMemcpyDeviceToHostAsync(result.sum_pow2.data(), sum_pow2, n_pixels * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(result.sum_pow3.data(), sum_pow3, n_pixels * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(result.sum_pow4.data(), sum_pow4, n_pixels * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(result.nonzero_channels_mask.data(), nonzero_channels, n_channels * sizeof(bool), stream);
    gpuStreamSynchronize(stream);
  }
  catch (...)
  {
    if (sum_pow2 != nullptr)
    {
      gpuFreeAsync(sum_pow2, stream);
    }
    if (sum_pow3 != nullptr)
    {
      gpuFreeAsync(sum_pow3, stream);
    }
    if (sum_pow4 != nullptr)
    {
      gpuFreeAsync(sum_pow4, stream);
    }
    throw;
  }

  gpuFreeAsync(sum_pow2, stream);
  gpuFreeAsync(sum_pow3, stream);
  gpuFreeAsync(sum_pow4, stream);
}
} // namespace breifast::gpu
