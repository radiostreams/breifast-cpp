#include "gpu_pipeline.hpp"

#include <stdexcept>

#include "gpu_device_utils.cuh"

namespace breifast::gpu
{
namespace
{
__global__ void combine_noise_map_rms_kernel(const float *rms, const float *rms_floor, const float *selfnoise_offset,
                                             const float *dde_noise, const float selfnoise_scale, const float dde_scale,
                                             float *out_noise_map, const size_t n_pixels)
{
  const auto i_pixel = global_index();
  if (i_pixel < n_pixels)
  {
    const auto base = fmaxf(rms[i_pixel], rms_floor[0]);
    out_noise_map[i_pixel] =
        sqrtf(base * base + selfnoise_scale * selfnoise_offset[i_pixel] + dde_scale * dde_noise[i_pixel]);
  }
}

__global__ void combine_noise_map_constant_kernel(const float *base_noise, const float *selfnoise_offset,
                                                  const float *dde_noise, const float selfnoise_scale,
                                                  const float dde_scale, float *out_noise_map, const size_t n_pixels)
{
  const auto i_pixel = global_index();
  if (i_pixel < n_pixels)
  {
    out_noise_map[i_pixel] = sqrtf(base_noise[0] * base_noise[0] + selfnoise_scale * selfnoise_offset[i_pixel] +
                                   dde_scale * dde_noise[i_pixel]);
  }
}
} // namespace

void gpu_combine_noise_map_rms(const float *rms, const float *rms_floor, const float *selfnoise_offset,
                               const float *dde_noise, const float selfnoise_scale, const float dde_scale,
                               float *out_noise_map, const size_t n_pixels, GpuStream stream)
{
  if (n_pixels == 0)
  {
    throw std::runtime_error("gpu_combine_noise_map_rms: zero pixels");
  }
  launch_per_element(combine_noise_map_rms_kernel, n_pixels, stream, rms, rms_floor, selfnoise_offset, dde_noise,
                     selfnoise_scale, dde_scale, out_noise_map, n_pixels);
}

void gpu_combine_noise_map_constant(const float *base_noise, const float *selfnoise_offset, const float *dde_noise,
                                    const float selfnoise_scale, const float dde_scale, float *out_noise_map,
                                    const size_t n_pixels, GpuStream stream)
{
  if (n_pixels == 0)
  {
    throw std::runtime_error("gpu_combine_noise_map_constant: zero pixels");
  }
  launch_per_element(combine_noise_map_constant_kernel, n_pixels, stream, base_noise, selfnoise_offset, dde_noise,
                     selfnoise_scale, dde_scale, out_noise_map, n_pixels);
}
} // namespace breifast::gpu