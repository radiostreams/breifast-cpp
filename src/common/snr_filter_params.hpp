#pragma once

namespace breifast::common
{

/// Scalars of the SNR and mask-cascade kernel, shared by the CPU oracle and the GPU twin.
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

} // namespace breifast::common
