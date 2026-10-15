#pragma once
// The SNR stage (themes 1, 3 and 4 of optimization_gpu/kernels.md) in one xsimd loop: SNR, the four cumulative masks,
// three histograms and the min/max sigma. Splitting it would add full passes to a loop tuned for bandwidth.
#include <array>
#include <cstddef>
#include <span>

namespace breifast::plane_lot::detail
{

/// Scalars of snr_and_filters().
struct SnrFilterParams
{
  float median_noise = 0.F;  ///< median of the noise map, scales `reject_noise_threshold`
  float inv_band_size = 1.F; ///< 1 / number of channels, averages the cutoff cubes
  long band_start = 0;       ///< channels `[band_start, band_stop)` the cutoff cubes are averaged over
  long band_stop = 1;
  long max_sigma_bin = 200;           ///< histograms have `2 * max_sigma_bin` bins, bin = floor(snr + max_sigma_bin)
  long histogram_detail = 0;          ///< > 0 fills histogram 2, > 1 fills histograms 0 and 1
  float reject_noise_threshold = 0.F; ///< 0 disables the noise rejection
  float threshold = 0.F;              ///< final mask: abs(snr) >= threshold
};

/// Min and max of the SNR over the pixels that pass the third mask; `+inf` and `lowest` when none does.
struct SnrFilterSummary
{
  float min_sigma;
  float max_sigma;
};

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
