#include "pointwise.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tracy/Tracy.hpp>

namespace breifast::plane_lot::detail
{
namespace
{
// Shared loop of the two combine_noise_map variants: `base_at(i)` is the base noise of pixel i.
template <typename BaseAt>
inline void combine_noise_map_loop(BaseAt base_at, std::span<const float> selfnoise_offset,
                                   std::span<const float> dde_noise, const float selfnoise_scale, const float dde_scale,
                                   std::span<float> out_noise_map)
{
  ZoneScopedN("combine");
  const size_t n_pixels = out_noise_map.size();
  if (selfnoise_offset.size() != n_pixels || dde_noise.size() != n_pixels)
  {
    throw std::runtime_error("combine_noise_map: selfnoise_offset or dde_noise size mismatch with out_noise_map");
  }

  float *const __restrict out = out_noise_map.data();
  const float *const __restrict selfnoise = selfnoise_offset.data();
  const float *const __restrict dde = dde_noise.data();
  for (size_t i_pixel = 0; i_pixel < n_pixels; ++i_pixel)
  {
    const float base = base_at(i_pixel);
    out[i_pixel] = std::sqrt(base * base + selfnoise_scale * selfnoise[i_pixel] + dde_scale * dde[i_pixel]);
  }
}
} // namespace

void combine_noise_map_rms(std::span<const float> rms, const float rms_floor, std::span<const float> selfnoise_offset,
                           std::span<const float> dde_noise, const float selfnoise_scale, const float dde_scale,
                           std::span<float> out_noise_map)
{
  if (rms.size() != out_noise_map.size())
  {
    throw std::runtime_error("combine_noise_map_rms: rms size mismatch with out_noise_map");
  }
  const float *const __restrict rms_ptr = rms.data();
  combine_noise_map_loop([&](size_t i_pixel) { return std::max(rms_ptr[i_pixel], rms_floor); }, selfnoise_offset,
                         dde_noise, selfnoise_scale, dde_scale, out_noise_map);
}

void combine_noise_map_constant(const float base_noise, std::span<const float> selfnoise_offset,
                                std::span<const float> dde_noise, const float selfnoise_scale, const float dde_scale,
                                std::span<float> out_noise_map)
{
  combine_noise_map_loop([&](size_t) { return base_noise; }, selfnoise_offset, dde_noise, selfnoise_scale, dde_scale,
                         out_noise_map);
}

} // namespace breifast::plane_lot::detail
