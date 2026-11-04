#pragma once
#include <cstddef>
#include <sstream>
#include <string>

#include "common/snr_filter_params.hpp"

// Host-side interface of the GPU library: no CUDA headers, so C++23 translation units can include it.
namespace breifast::gpu
{
struct GpuDeviceSummary
{
  std::string name;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  size_t total_global_memory = 0;
  int multiprocessor_count = 0;
  int max_threads_per_block = 0;
  int max_threads_per_multiprocessor = 0;

  std::string to_string() const
  {
    std::stringstream ss;
    ss << "GPU Device: " << name << "\n";
    ss << "  Compute capability: " << compute_capability_major << "." << compute_capability_minor << "\n";
    ss << "  Total global memory: " << (total_global_memory / (1024 * 1024)) << " MiB\n";
    ss << "  Multiprocessors: " << multiprocessor_count << "\n";
    ss << "  Max threads per block: " << max_threads_per_block << "\n";
    ss << "  Max threads per multiprocessor: " << max_threads_per_multiprocessor << "\n";
    return ss.str();
  }
};

/// Properties of the GPU with index `device_id` among the devices visible to the process.
GpuDeviceSummary gpu_device_summary(int device_id = 0);

/// The stream a GPU call runs on; `nullptr` is the default stream. Opaque here so the header needs no CUDA includes,
/// and implicitly a `cudaStream_t` inside CUDA translation units.
struct CUstream_st;
struct GpuStream
{
  void *handle_ = nullptr;
  GpuStream(std::nullptr_t = nullptr)
  {
  }
  GpuStream(void *h) : handle_(h)
  {
  }
#ifdef __CUDACC__
  operator ::CUstream_st *() const
  {
    return static_cast<::CUstream_st *>(handle_);
  }
#endif
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
void gpu_mask_or(const bool *accept_mask, size_t n_pixels, int band_start, int band_stop, bool *out_mask, int *out_any,
                 GpuStream stream = nullptr);

/**
 * @brief Weighted band image: `out[p] = Σ_c cube[c,p]*weights[c]*inv_weights_sum - Σ_c mean[c,p]*inv_band_size`.
 *
 * Device pointers. `mean_image` may be `nullptr`, then the second sum is skipped. Work is enqueued on `stream`.
 * @param cube `[n_channels, n_pixels]` floats, channel major.
 * @param weights One weight per channel, indexed by the absolute channel.
 * @param mean_image `[n_channels, n_pixels]` floats, or `nullptr`.
 * @param out_plane `[n_pixels]` floats, written completely.
 */
void gpu_weighted_band_image(const float *cube, const float *weights, const float *mean_image, float *out_plane,
                             size_t n_pixels, int band_start, int band_stop, float inv_weights_sum, float inv_band_size,
                             GpuStream stream = nullptr);

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
                                   float *out_sum4, size_t n_pixels, int n_channels, GpuStream stream = nullptr);

void gpu_combine_noise_map_rms(const float *rms, float rms_floor, const float *selfnoise_offset, const float *dde_noise,
                               float selfnoise_scale, float dde_scale, float *out_noise_map, size_t n_pixels,
                               GpuStream stream = nullptr);

void gpu_combine_noise_map_constant(float base_noise, const float *selfnoise_offset, const float *dde_noise,
                                    float selfnoise_scale, float dde_scale, float *out_noise_map, size_t n_pixels,
                                    GpuStream stream = nullptr);

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
 * @param params scalars, see `SnrFilterParams`; `max_sigma_bin` sets the histogram length `2 * max_sigma_bin`.
 * @param out_snr `[n_pixels]` floats, written completely.
 * @param image_mask `[n_pixels]` bools, AND-ed in place, written completely.
 * @param histogram0 `2 * max_sigma_bin` counts under the first mask (only when `histogram_detail > 1`), and likewise
 *        `histogram1` under the second mask; `histogram2` under the third mask when `histogram_detail > 0`.
 * @param out_min_sigma,out_max_sigma One float each; `+inf` and `lowest` when no pixel passes the third mask.
 */
void gpu_snr_and_filters(const float *plane, const float *noise_map, const float *snr_cutoff, const float *flux_cutoff,
                         size_t n_pixels, const common::SnrFilterParams &params, float *out_snr, bool *image_mask,
                         unsigned long long *histogram0, unsigned long long *histogram1, unsigned long long *histogram2,
                         float *out_min_sigma, float *out_max_sigma, GpuStream stream = nullptr);

/**
 * @brief Local RMS noise map: a sliding `boxsize x boxsize` window over `plane`, edge-corrected, square root of the
 * mean of `plane^2`.
 *
 * Two separable passes with double accumulators, the horizontal running sums into `row_window_sums` then the vertical
 * pass. Device pointers; the work is enqueued on `stream`. `boxsize <= 0` is treated as 1.
 * @param plane `[n_y, n_x]` floats, row major.
 * @param row_window_sums Caller scratch of `n_y * n_x` floats, overwritten.
 * @param out_rms `[n_y, n_x]` floats, written completely.
 */
void gpu_rms_noise_map(const float *plane, size_t n_y, size_t n_x, long boxsize, float *row_window_sums, float *out_rms,
                       GpuStream stream = nullptr);

/**
 * @brief Exact median of `values`, numpy semantics: odd `n` picks rank `n/2`, even `n` averages ranks `n/2 - 1` and
 *        `n/2`.
 *
 * Order-preserving key, four MSB-first byte passes with a device-wide 256-bin digit histogram, then a scan of the bins
 * per pass. Device pointers; the work is enqueued on `stream`. `n == 0` writes NaN.
 * @param values `[n]` floats, read only.
 * @param out_median One float, written by this call.
 */
void gpu_median(const float *values, size_t n, float *out_median, GpuStream stream = nullptr);

/**
 * @brief Exact median of `|values|`, the global MAD input.
 *
 * Same as gpu_median() on the absolute values.
 * @param values `[n]` floats, read only.
 * @param out_median One float, written by this call.
 */
void gpu_median_abs(const float *values, size_t n, float *out_median, GpuStream stream = nullptr);

/**
 * @brief Scaled median absolute value of the nonzero pixels of each stamp: the `lmad` of stamp_donut_stats.
 *
 * One block per stamp. The block keeps the pixels with `value != 0`, takes the median of `abs(value)` with a
 * block-level digit select (the even-count average of the two middle ranks), and writes `scale * median`, or 0 when
 * the stamp has no nonzero pixel. Device pointers; the work is enqueued on `stream`.
 * @param stamps `[n_stamps, stamp_pixels]` floats, row major, one stamp per row.
 * @param stamp_pixels `(2 * radius + 1)^2`.
 * @param scale Multiplier applied to the median, `1.4826` for the donut.
 * @param out_lmad `[n_stamps]` floats, written completely.
 */
void gpu_donut_median_abs(const float *stamps, size_t n_stamps, size_t stamp_pixels, float scale, float *out_lmad,
                          GpuStream stream = nullptr);

/**
 * @brief `max(|min|, |max|)` over each stamp, the local-noise extreme of stamp_abs_extreme.
 *
 * One block per stamp, min and max reduced over the stamp and combined into one number per stamp. NaN pixels are
 * ignored, and a stamp with no other value gives `FLT_MAX`. Device pointers; the work is enqueued on `stream`.
 * @param stamps `[n_stamps, stamp_pixels]` floats, row major, one stamp per row.
 * @param stamp_pixels `(2 * radius + 1)^2`.
 * @param out_extreme `[n_stamps]` floats, written completely.
 */
void gpu_stamp_abs_extreme(const float *stamps, size_t n_stamps, size_t stamp_pixels, float *out_extreme,
                           GpuStream stream = nullptr);
} // namespace breifast::gpu
