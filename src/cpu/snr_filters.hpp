#pragma once
#include <array>
#include <cstddef>
#include <span>

#include "common/snr_filter_params.hpp"

namespace breifast::plane_lot::detail
{

using common::SnrFilterParams;
using common::SnrFilterSummary;

/**
 * @brief SNR and mask cascade of one band (breifast.py:741-804).
 *
 * `out_snr = plane / noise_map` where the noise is nonzero and finite, else 0. `image_mask` is refined in place through
 * four cumulative stages per pixel: valid noise, `abs(snr) >= mean of snr_cutoff over the band` (when given),
 * `abs(plane) >= mean of flux_cutoff over the band` (when given), `noise_map <= median_noise * reject_noise_threshold`
 * (when the threshold is nonzero); the final value is `abs(snr) >= threshold` of the pixels that passed. The histograms
 * count `snr` under the masks of stages 0, 1 and 2. Writes `out_snr` and the histograms completely.
 * @param snr_cutoff `[n_channels, n_pixels]`, or empty for no cutoff.
 * @param flux_cutoff `[n_channels, n_pixels]`, or empty for no cutoff.
 * @param histograms Three spans of `2 * max_sigma_bin` counts.
 */
SnrFilterSummary snr_and_filters(std::span<const float> plane, std::span<const float> noise_map,
                                 std::span<const float> snr_cutoff, std::span<const float> flux_cutoff,
                                 const SnrFilterParams &params, std::span<float> out_snr, std::span<bool> image_mask,
                                 std::array<std::span<long>, 3> histograms);

} // namespace breifast::plane_lot::detail
