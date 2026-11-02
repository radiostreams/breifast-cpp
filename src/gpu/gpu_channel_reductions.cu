#include <stdexcept>

#include "gpu_pipeline.hpp"
#include "gpu_runtime.hpp"

namespace breifast::gpu
{
namespace
{
// TODO: out_mask[p] = OR over c in [band_start, band_stop) of accept_mask[c * n_pixels + p], and *out_any set when
// any out_mask[p] is true.
__global__ void mask_or_kernel(const bool *accept_mask, size_t n_pixels, int band_start, int band_stop, bool *out_mask,
                               int *out_any)
{
  const auto i_pixel = blockIdx.x * blockDim.x + threadIdx.x;
  bool accept = false;
  if (i_pixel < n_pixels)
  {
    for (int i_channel = band_start; i_channel < band_stop; ++i_channel)
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
                                           float *out_plane, size_t n_pixels, int band_start, int band_stop,
                                           float inv_weights_sum, float inv_band_size)
{
  const auto i_pixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_pixel < n_pixels)
  {
    float result = 0.0f;
    for (int i_channel = band_start; i_channel < band_stop; ++i_channel)
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
} // namespace

void gpu_mask_or(const bool *accept_mask, size_t n_pixels, int band_start, int band_stop, bool *out_mask, int *out_any,
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
                             size_t n_pixels, int band_start, int band_stop, float inv_weights_sum, float inv_band_size,
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
} // namespace breifast::gpu
