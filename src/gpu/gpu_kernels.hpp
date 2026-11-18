#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/snr_filter_params.hpp"
#include "gpu_runtime.hpp"

// Host-side interface of the GPU kernels: the launch wrappers and the scratch they reuse.
namespace breifast::gpu
{
/// Initial label of a pixel outside the mask, before gpu_label_islands converges; a pixel inside starts at its flat
/// index.
constexpr unsigned int background_label = ~0U;

/// Device scratch of gpu_median and gpu_median_abs: the digit histogram, then the radix-select state (count below the
/// prefix, prefix, coverage), the largest key below the prefix and the count of blocks done with the current pass.
struct MedianScratch
{
  static constexpr std::size_t histogram_bins = 2048;
  DeviceVector<std::uint32_t> words{histogram_bins + 5};
};

/// Device scratch of gpu_label_islands: the per-pixel root ranks, the convergence flag, and the temporary storage of
/// the rank scan (cub::DeviceScan).
struct LabelScratch
{
  /// Grows every array for a plane of `n_pixels`; the contents are unspecified after growing.
  void ensure(std::size_t n_pixels); // in gpu_islands.cu, where CUB is available

  DeviceVector<unsigned int> root_ranks;
  DeviceVector<unsigned int> changed{1};
  DeviceVector<std::byte> scan_storage;
};

/// Device scratch of gpu_label_stats and gpu_island_boost: the per-island reduction keys and partial results.
struct IslandScratch
{
  /// Grows every array to hold `n_islands` islands; the contents are unspecified after growing.
  void ensure(std::size_t n_islands)
  {
    value_keys.ensure(n_islands);
    local_noise_max.ensure(n_islands);
    sizes.ensure(n_islands);
    boost_keys.ensure(n_islands + 1); // indexed by label, 1..n_islands
  }

  DeviceVector<unsigned long long> value_keys;
  DeviceVector<float> local_noise_max;
  DeviceVector<unsigned int> sizes;
  DeviceVector<unsigned long long> boost_keys;
};

/**
 * @brief OR of the accept masks of the channels `[band_start, band_stop)`, and whether any pixel ends up true.
 *
 * All pointers are device pointers. The work is enqueued on `stream` and the call returns before it finishes.
 * @param accept_mask `[n_channels, n_pixels]` bools, channel major.
 * @param n_pixels `Y * X` of one channel.
 * @param band_start First channel of the band.
 * @param band_stop One past the last channel of the band, `band_start < band_stop`.
 * @param out_mask `[n_pixels]` bools, true where any channel of the band is true.
 * @param out_any One int, nonzero when any pixel of `out_mask` is true. Written by this call, no host read-back.
 */
void gpu_mask_or(const bool *accept_mask, size_t n_pixels, long band_start, long band_stop, bool *out_mask,
                 int *out_any, cudaStream_t stream = nullptr);

/**
 * @brief Weighted band image: `out[p] = Σ_c cube[c,p]*weights[c]*inv_weights_sum - Σ_c mean[c,p]*inv_band_size`.
 *
 * `inv_weights_sum = 1 / Σ_c weights[c]` over the band, summed by the kernel in channel order like the host
 * std::accumulate. Also gpu_median_abs() of the band image, times `scale`: the band image's kernel is its first pass.
 * Device pointers. `mean_image` may be `nullptr`, then the second sum is skipped. Work is enqueued on `stream`.
 * @param cube `[n_channels, n_pixels]` floats, channel major.
 * @param weights One weight per channel, indexed by the absolute channel.
 * @param mean_image `[n_channels, n_pixels]` floats, or `nullptr`.
 * @param out_plane `[n_pixels]` floats, written completely.
 * @param out_median_abs One device float.
 */
void gpu_weighted_band_image(const float *cube, const float *weights, const float *mean_image, float *out_plane,
                             size_t n_pixels, long band_start, long band_stop, float inv_band_size,
                             float *out_median_abs, float scale, MedianScratch &scratch, cudaStream_t stream = nullptr);

/**
 * @brief Per-pixel sums of the second, third and fourth powers over the valid channels:
 *        `out_sum_k[p] = Σ over channels c with nonzero_channels[c] of cube[c, p]^k`, k = 2, 3, 4.
 *
 * Device pointers. Channels flagged false are skipped without being read. The work is enqueued on `stream`.
 * @param cube `[n_channels, n_pixels]` floats, channel major.
 * @param nonzero_channels One flag per channel, indexed by the absolute channel.
 * @param n_pixels `Y * X` of one channel.
 * @param out_sum2 `[n_pixels]` floats, written completely. Same for `out_sum3` and `out_sum4`.
 */
void gpu_spectral_variance_moments(const float *cube, const bool *nonzero_channels, float *out_sum2, float *out_sum3,
                                   float *out_sum4, size_t n_pixels, int n_channels, cudaStream_t stream = nullptr);

/**
 * @brief Noise map: `out[p] = sqrt(base[p]^2 + selfnoise_scale * selfnoise_offset[p] + dde_scale * dde_noise[p])`.
 *
 * `base[p] = max(rms[p], *base_noise)` with an RMS map, where `base_noise` is the RMS floor; `rms == nullptr` gives
 * `base[p] = *base_noise` for every pixel. Also the median of the noise map, as gpu_median() computes it: the noise
 * map's kernel is its first pass. Device pointers; the work is enqueued on `stream`.
 * @param rms `[n_pixels]` floats, or `nullptr`.
 * @param base_noise One device float.
 * @param out_noise_map `[n_pixels]` floats, written completely.
 * @param out_median One device float.
 */
void gpu_combine_noise_map(const float *rms, const float *base_noise, const float *selfnoise_offset,
                           const float *dde_noise, float selfnoise_scale, float dde_scale, float *out_noise_map,
                           size_t n_pixels, float *out_median, MedianScratch &scratch, cudaStream_t stream = nullptr);

/**
 * @brief SNR and mask cascade of one band, fused: `snr`, the four cumulative masks, the histogram counts and the SNR
 *        min/max over the accepted pixels, in one pass over the plane.
 *
 * `out_snr[p] = plane[p] / noise_map[p]` where the noise is nonzero and finite, else 0. `image_mask` is refined in
 * place through four cumulative stages per pixel: valid noise, `|snr| >= mean of snr_cutoff over the band` (when
 * given),
 * `|plane| >= mean of flux_cutoff over the band` (when given), `noise_map <= median_noise * reject_noise_threshold`
 * (when the threshold is nonzero); the final value is `|snr| >= threshold` of the pixels that passed. Device pointers;
 * `snr_cutoff` and `flux_cutoff` may be `nullptr` for no cutoff, and are indexed `[c * n_pixels + p]` for `c` in
 * `[band_start, band_stop)`. The work is enqueued on `stream`; the summary lands in `out_min_sigma`/`out_max_sigma`.
 * @param median_noise One device float, the noise-map median that scales `reject_noise_threshold`.
 * @param params scalars, see `SnrFilterParams`; `max_sigma_bin` sets the histogram length `2 * max_sigma_bin`.
 *        The GPU path takes `median_noise` from the device pointer, not from `params.median_noise`.
 * @param out_snr `[n_pixels]` floats, written completely.
 * @param image_mask `[n_pixels]` bools, AND-ed in place, written completely.
 * @param histogram0 `2 * max_sigma_bin` counts under the first mask (only when `histogram_detail > 1`), and likewise
 *        `histogram1` under the second mask; `histogram2` under the third mask when `histogram_detail > 0`. Every
 *        non-null histogram is zeroed on every call, including the ones this band does not fill, matching the CPU.
 * @param out_min_sigma,out_max_sigma One float each; `+inf` and `lowest` when no pixel passes the third mask.
 * @param labels `[n_pixels]`, written completely: the initial labels of gpu_label_islands for `image_mask`.
 */
void gpu_snr_and_filters(const float *plane, const float *noise_map, const float *snr_cutoff, const float *flux_cutoff,
                         const float *median_noise, size_t n_pixels, const common::SnrFilterParams &params,
                         float *out_snr, const bool *accept_mask, bool *image_mask, unsigned long long *histogram0,
                         unsigned long long *histogram1, unsigned long long *histogram2, float *out_min_sigma,
                         float *out_max_sigma, unsigned int *labels, cudaStream_t stream = nullptr);

/**
 * @brief Local RMS noise map: a sliding `boxsize x boxsize` window over `plane`, edge-corrected, square root of the
 * mean of `plane^2`.
 *
 * Two separable passes with double accumulators, the horizontal running sums into `row_window_sums` then the vertical
 * pass. Also gpu_median() of the RMS map: the vertical pass is its first pass. Device pointers; the work is enqueued on
 * `stream`. `boxsize <= 0` is treated as 1.
 * @param plane `[n_y, n_x]` floats, row major.
 * @param row_window_sums Caller scratch of `n_y * n_x` floats, overwritten.
 * @param row_prefix Caller scratch of `n_y * n_x` doubles, overwritten: the per-row prefix sums of `plane^2`.
 * @param row_prefix_scan_storage Caller scratch for the scan's temporary storage; grown as needed.
 * @param out_rms `[n_y, n_x]` floats, written completely.
 * @param out_median One device float.
 */
void gpu_rms_noise_map(const float *plane, size_t n_y, size_t n_x, long boxsize, float *row_window_sums,
                       double *row_prefix, DeviceVector<std::byte> &row_prefix_scan_storage, float *out_rms,
                       float *out_median, MedianScratch &scratch, cudaStream_t stream = nullptr);

/**
 * @brief Exact median of `values`, numpy semantics: odd `n` picks rank `n/2`, even `n` averages ranks `n/2 - 1` and
 *        `n/2`.
 *
 * Order-preserving key, four MSB-first byte passes with a device-wide 256-bin digit histogram, then a scan of the bins
 * per pass. Device pointers; the work is enqueued on `stream`. `n` must be greater than 0.
 * @param values `[n]` floats, read only.
 * @param out_median One float, written by this call.
 * @param scratch Reused by every call on the same stream; each call resets the parts it reads.
 */
void gpu_median(const float *values, size_t n, float *out_median, MedianScratch &scratch,
                cudaStream_t stream = nullptr);

/**
 * @brief Exact median of `|values|`, the global MAD input.
 *
 * Same as gpu_median() on the absolute values.
 * @param values `[n]` floats, read only.
 * @param out_median One float, written by this call.
 * @param scale Multiplier applied to the median before writing it.
 */
void gpu_median_abs(const float *values, size_t n, float *out_median, float scale, MedianScratch &scratch,
                    cudaStream_t stream = nullptr);

/// Where each island's stamp is read from: the (2 * radius + 1)^2 window of `plane` centred on the island's peak, read
/// in place (device Stamp, gpu_device_utils.cuh). A cell outside the plane reads 0, and so does a cell of the central
/// (2 * inner_half + 1)^2 box when inner_half >= 0 (clipped to radius); inner_half < 0 keeps the centre.
struct StampWindows
{
  const float *plane = nullptr; ///< [n_y, n_x], row major
  size_t n_y = 0;
  size_t n_x = 0;
  const long *peak_y = nullptr; ///< [n_islands]
  const long *peak_x = nullptr; ///< [n_islands]
  long radius = 0;
  long inner_half = -1;
};

/**
 * @brief Scaled median absolute value of the nonzero pixels of each stamp: the `lmad` of stamp_donut_stats.
 *
 * One block per stamp. The block keeps the pixels with `value != 0`, takes the median of `abs(value)` with a
 * block-level digit select (the even-count average of the two middle ranks), and writes `scale * median`, or 0 when
 * the stamp has no nonzero pixel. Device pointers; the work is enqueued on `stream`.
 * @param stamps The stamp of each island.
 * @param n_islands Number of islands, at least 1.
 * @param scale Multiplier applied to the median, `1.4826` for the donut.
 * @param out_lmad `[n_islands]` floats, written.
 */
void gpu_donut_median_abs(const StampWindows &stamps, unsigned int n_islands, float scale, float *out_lmad,
                          cudaStream_t stream = nullptr);

/**
 * @brief Population standard deviation of the nonzero pixels of each stamp: the `stddev` of stamp_donut_stats.
 *
 * One block per stamp. Over the nonzero pixels (count `m`) it reduces `sum` and `sumsq` in double and writes
 * `sqrt(sumsq/m - (sum/m)^2)`, or 0 when the stamp has no nonzero pixel. Population (divide by `m`), matching
 * `xt::stddev`. Device pointers; the work is enqueued on `stream`.
 * @param stamps The stamp of each island.
 * @param n_islands Number of islands, at least 1.
 * @param out_std `[n_islands]` floats, written.
 */
void gpu_donut_std(const StampWindows &stamps, unsigned int n_islands, float *out_std, cudaStream_t stream = nullptr);

/**
 * @brief `max(|min|, |max|)` over each stamp, the local-noise extreme of stamp_abs_extreme.
 *
 * One block per stamp, min and max reduced over the stamp and combined into one number per stamp. NaN pixels are
 * ignored, and a stamp with no other value gives `FLT_MAX`. Device pointers; the work is enqueued on `stream`.
 * @param stamps The stamp of each island.
 * @param n_islands Number of islands, at least 1.
 * @param out_extreme `[n_islands]` floats, written.
 */
void gpu_stamp_abs_extreme(const StampWindows &stamps, unsigned int n_islands, float *out_extreme,
                           cudaStream_t stream = nullptr);

/**
 * @brief Reduced chi2 of each stamp against the beam kernel, over the nonzero stamp pixels.
 *
 * One block per stamp. `scale = sum(stamp) / kernel_sum` over the whole stamp, then
 * `sum((stamp[i] - scale * kernel[i])^2) / (adjacent_noise^2 * count)` over the nonzero pixels, and 0 when there
 * is no nonzero pixel or `adjacent_noise` is 0. `adjacent_noise` is computed on the fly as
 * `stamp_extreme[i] * inv_max_to_std_ratio`. Device pointers; the work is enqueued on `stream`.
 * @param stamps The beam stamp of each island, `radius` the beam halfsize.
 * @param kernel `[(2 * radius + 1)^2]` floats, the band's beam kernel, shared by every stamp.
 * @param kernel_sum Sum of `kernel`, computed on the host (beam_kernel_sum).
 * @param n_islands Number of islands, at least 1.
 * @param stamp_extreme `[n_islands]` floats, the per-island `stamp_abs_extreme` values.
 * @param inv_max_to_std_ratio `1 / max_to_std_ratio_value`.
 * @param out_chi2 `[n_islands]` floats, written.
 */
void gpu_stamp_beam_chi2(const StampWindows &stamps, const float *kernel, float kernel_sum, unsigned int n_islands,
                         const float *stamp_extreme, float inv_max_to_std_ratio, float *out_chi2,
                         cudaStream_t stream = nullptr);

/**
 * @brief The values of four maps at each island's peak: the `radius = 0` reads of the detection record.
 *
 * One thread per island. Device pointers; the work is enqueued on `stream`.
 * @param plane,noise_map,selfnoise_raw `[n_y, n_x]` floats, row major.
 * @param rms_map `[n_y, n_x]` floats, or null in the global MAD mode, which writes 0.
 * @param n_x Row width.
 * @param peak_y,peak_x `[n_islands]` peak row and column per island, inside the maps.
 * @param n_islands Number of islands, at least 1.
 * @param flux,noise_at_peak,rms_at_peak,selfnoise_at_peak `[n_islands]` floats, written.
 */
void gpu_gather_at_peaks(const float *plane, const float *noise_map, const float *rms_map, const float *selfnoise_raw,
                         size_t n_x, const long *peak_y, const long *peak_x, unsigned int n_islands, float *flux,
                         float *noise_at_peak, float *rms_at_peak, float *selfnoise_at_peak,
                         cudaStream_t stream = nullptr);

/**
 * @brief 4-connected component labels of `mask`, numbered `1..count` in raster order of each island's first pixel.
 *
 * A dense min-label propagation: every true pixel starts at its flat index, then repeatedly takes the smallest of its
 * flat-index label and its four already-true neighbours until a fixed point, so each pixel settles on its component's
 * smallest index. That index is then ranked among the components. `0` marks background. Device pointers; the work is
 * enqueued on `stream` and converges on the device with a cooperative grid sync, so this call does not synchronise
 * `stream`. Requires cooperative-launch support. `n_y * n_x` must be positive.
 * @param mask `[n_y * n_x]` bools, row major.
 * @param labels `[n_y * n_x]`: on entry the initial labels of `mask` (from gpu_snr_and_filters or gpu_init_labels), on
 *        exit the island labels, `0` outside every island.
 * @param out_count One `unsigned int`, the number of islands.
 * @param scratch Grown to at least `n_y * n_x` pixels.
 */
void gpu_label_islands(const bool *mask, size_t n_y, size_t n_x, unsigned int *labels, unsigned int *out_count,
                       LabelScratch &scratch, cudaStream_t stream = nullptr);

/// Initial labels of gpu_label_islands for `mask`: the flat index where `mask` is set, background_label elsewhere.
void gpu_init_labels(const bool *mask, size_t n, unsigned int *labels, cudaStream_t stream = nullptr);

/**
 * @brief Per-label statistics of a labeled image: the `peak_per_label` outputs of the CPU label_islands() pass.
 *
 * For every label `1..n_labels`, over the pixels carrying that label:
 * - `island_size`: pixel count.
 * - `value`: largest `|image[p]|`, with the first pixel in raster order winning a tie.
 * - `peak_y`, `peak_x`: coordinates of that winning pixel.
 * - `out_local_noise`: largest `local_noise[p]`.
 *
 * Device pointers. `labels` is compact, `0` for background, `1..n_labels` for islands. The work is enqueued on
 * `stream`. Every output is written for every label. A label with no pixel gets `value = -FLT_MAX`,
 * `out_local_noise = -FLT_MAX`, `peak_y = peak_x = 0`, `island_size = 0`.
 * @param labels `[n_pixels]` compact labels, row major.
 * @param image `[n_pixels]` floats whose absolute maximum defines the peak.
 * @param local_noise `[n_pixels]` floats.
 * @param n_x Row width, used to turn the peak flat index into `(peak_y, peak_x)`.
 * @param n_labels Number of labels, at least 1.
 * @param value `[n_labels]` floats, written.
 * @param out_local_noise `[n_labels]` floats, written.
 * @param peak_y `[n_labels]` peak rows, written.
 * @param peak_x `[n_labels]` peak columns, written.
 * @param island_size `[n_labels]` pixel counts, written.
 * @param scratch Grown to at least `n_labels` islands.
 */
void gpu_label_stats(const unsigned int *labels, const float *image, const float *local_noise, size_t n_pixels,
                     size_t n_x, unsigned int n_labels, float *value, float *out_local_noise, long *peak_y,
                     long *peak_x, unsigned int *island_size, IslandScratch &scratch, cudaStream_t stream = nullptr);

/**
 * @brief Batched island boost: for every compact label `1..n_labels`, the largest-magnitude
 *        `flux_per_label[label - 1] / baseline_flux[p]` over the pixels carrying that label, with the first pixel in
 *        raster order winning a tie.
 *
 * Device pointers. The work is enqueued on `stream`.
 * @param labels_per_pixel `[n_pixels]` labels, row major.
 * @param baseline_flux `[n_pixels]` floats, row major.
 * @param n_pixels Number of pixels.
 * @param n_labels Number of labels, at least 1.
 * @param flux_per_label `[n_labels]` floats, one numerator per label.
 * @param boost `[n_labels]` floats, written; 0 when a label has no pixel.
 * @param scratch Grown to at least `n_labels` islands.
 */
void gpu_island_boost(const unsigned int *labels_per_pixel, const float *baseline_flux, size_t n_pixels,
                      unsigned int n_labels, const float *flux_per_label, float *boost, IslandScratch &scratch,
                      cudaStream_t stream = nullptr);
} // namespace breifast::gpu
