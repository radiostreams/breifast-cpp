#pragma once
#include <cstddef>
#include <span>

namespace breifast::plane_lot::detail
{

/**
 * @brief Noise map from a local RMS map: `out = sqrt(max(rms, rms_floor)^2 + selfnoise_scale * selfnoise_offset +
 *        dde_scale * dde_noise)`. Pixels of `rms` below `rms_floor` are raised to it. Writes `out_noise_map`
 *        completely; every span holds one value per pixel.
 */
void combine_noise_map_rms(std::span<const float> rms, const float rms_floor, std::span<const float> selfnoise_offset,
                           std::span<const float> dde_noise, const float selfnoise_scale, const float dde_scale,
                           std::span<float> out_noise_map);

/**
 * @brief Noise map from one base noise for every pixel (the global MAD): `out = sqrt(base^2 + selfnoise_scale *
 *        selfnoise_offset + dde_scale * dde_noise)`. Writes `out_noise_map` completely.
 */
void combine_noise_map_constant(const float base_noise, std::span<const float> selfnoise_offset,
                                std::span<const float> dde_noise, const float selfnoise_scale, const float dde_scale,
                                std::span<float> out_noise_map);

} // namespace breifast::plane_lot::detail
