#include "gpu_pipeline.hpp"

#include <stdexcept>

#include "gpu_device_utils.cuh"

namespace breifast::gpu
{
namespace
{
// The base noise is max(rms[p], *base_noise) with an RMS map, else *base_noise.
__global__ void combine_noise_map_kernel(const float *rms, const float *base_noise, const float *selfnoise_offset,
                                         const float *dde_noise, const float selfnoise_scale, const float dde_scale,
                                         float *out_noise_map, const size_t n_pixels)
{
  const auto i_pixel = global_index();
  if (i_pixel < n_pixels)
  {
    const float base = rms != nullptr ? fmaxf(rms[i_pixel], *base_noise) : *base_noise;
    out_noise_map[i_pixel] =
        sqrtf(base * base + selfnoise_scale * selfnoise_offset[i_pixel] + dde_scale * dde_noise[i_pixel]);
  }
}
} // namespace

void gpu_combine_noise_map(const float *rms, const float *base_noise, const float *selfnoise_offset,
                           const float *dde_noise, const float selfnoise_scale, const float dde_scale,
                           float *out_noise_map, const size_t n_pixels, GpuStream stream)
{
  if (n_pixels == 0)
  {
    throw std::runtime_error("gpu_combine_noise_map: zero pixels");
  }
  launch_per_element(combine_noise_map_kernel, n_pixels, stream, rms, base_noise, selfnoise_offset, dde_noise,
                     selfnoise_scale, dde_scale, out_noise_map, n_pixels);
}
} // namespace breifast::gpu
