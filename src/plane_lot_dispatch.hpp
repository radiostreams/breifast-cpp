/**
 * @file plane_lot_dispatch.hpp
 * @brief Manual C++ reimplementation of breifast.py's per-lot plane-processing dispatch (breifast.py:1157-1197).
 *
 * Pipeline shape: a `LotBatch` holds one `PlaneBatchItem` per (interval, stokes) pair plus the shared
 * `PlaneBatchGroup` arrays each item's group_index points at. `run_lot_core()`/`run_batch_core()` dispatch
 * `process_plane()` over all items in parallel (via TBB); `process_plane()` loops over `PlaneConfig::band_slices`,
 * calling `process_band()` for each; `process_band()` computes the band's weighted image, noise map, SNR, and
 * (eventually) per-island detections into a shared `ProcessPlaneResult&`.
 *
 * Most numeric steps have two implementations selected by the `USE_XTENSOR` preprocessor macro: an
 * xtensor-expression version and a manual-loop version, kept side by side so the faster one can be benchmarked and
 * the other dropped later. Axis convention throughout: axis 0 of every 2D plane/image array is Y (rows), axis 1 is
 * X (columns), matching breifast.py's own xaxis/yaxis definitions -- array accesses are consistently written
 * array(i_y, i_x).
 */
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <vector>

#include <oneapi/tbb/concurrent_unordered_map.h>
#include <oneapi/tbb/mutex.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xshape.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/reducers/xreducer.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>

namespace breifast
{
/**
 * @brief One contiguous range of channels within a plane, processed as one unit.
 *
 * Iterable (`begin()`/`end()`) over its channel indices via `std::views::iota`. `xrange()` returns a fresh
 * `xt::range(start, stop)` on every call — xtensor's range adaptor must be constructed at each `xt::view()`
 * call site, not stored and reused, or stride computation breaks.
 */
struct BandSlice
{
  long start = std::numeric_limits<long>::max();
  long stop = std::numeric_limits<long>::lowest();
  /// Precomputed self-noise scaling factor for this band (Python's `snsf` = `full_bandwidth / bandwidth`),
  /// supplied by the Python side rather than computed here.
  float selfnoise_scale_factor = std::numeric_limits<float>::quiet_NaN();
  std::string label;

  auto begin() const
  {
    return std::views::iota(start, stop).begin();
  }
  auto end() const
  {
    return std::views::iota(start, stop).end();
  }

  auto cbegin() const
  {
    return std::views::iota(start, stop).begin();
  }
  auto cend() const
  {
    return std::views::iota(start, stop).end();
  }

  size_t size() const
  {
    return stop - start;
  }

  auto xrange() const
  {
    return xt::range(start, stop);
  }
};

/**
 * @brief Run-level configuration shared by every plane/band processed in a lot.
 *
 * Mirrors the keyword arguments of breifast.py's per-plane processing loop (subtract_mean, noise-estimator choice,
 * cutoff/threshold parameters, etc.).
 */
class PlaneConfig
{
private:
  /**
   * @brief Median of the max of `n` iid standard-normal samples (breifast.py:67-74's `max_to_std_ratio`).
   *
   * Finds where the CDF of "max of n samples" crosses 0.5 via a 1000-point search over `[-10, 10]`, linearly
   * interpolating between the two straddling samples. Result is cached in a process-wide map (shared by every
   * `PlaneConfig` instance), keyed by `n`, so it's computed at most once per distinct `n`.
   * @param n Sample count (typically a postage-stamp box's pixel count).
   * @return The ratio, always non-negative.
   */
  static float max_to_std_ratio(const size_t n);

  mutable std::optional<float> max_to_std_ratio_value_;

public:
  std::vector<BandSlice> band_slices;
  bool subtract_mean = false;
  bool use_rms_noise_estimator = true;
  long boxsize = 0;
  long inner_boxsize = 10;
  float ddenoise_scaling_factor = 1.f;
  float full_bandwidth = 0.f;
  float reject_noise_threshold = 0.f;
  long histogram_detail = 0;
  long max_sigma_bin = 200;
  float threshold = 0.f;
  bool save_stat_cubes = false;
  bool beam_size_enabled = false;
  std::optional<xt::xarray<double>> fixed_beam_kernel;
  long fixed_beam_halfsize = 0;
  std::optional<xt::xarray<float>> baseline_flux;

  /**
   * @brief max_to_std_ratio() for this config's postage-stamp box (`(2*(boxsize/2)+1)^2` pixels), cached per instance
   *        after the first call.
   * @return The cached ratio.
   */
  float max_to_std_ratio_value() const;
};

/**
 * @brief One (interval, stokes) unit of work: a plane's worth of channel data.
 *
 * `group_index` looks up the shared `PlaneBatchGroup` (accept_mask, mean_image, noise arrays) this item's Stokes
 * label reuses across many planes in the lot.
 */
struct PlaneBatchItem
{
  size_t group_index = 0;
  xt::xarray<float> stokes_cube;
  xt::xarray<float> weights;
  xt::xarray<bool> nonzero_channels;
  std::string stokes_label;
  std::optional<xt::xarray<float>> beam_major_axis, beam_minor_axis, beam_position_angle;
};

/**
 * @brief Per-Stokes-label arrays shared by every `PlaneBatchItem` that references them via `group_index` (accept
 *        mask, mean image, noise/cutoff maps).
 */
struct PlaneBatchGroup
{
  xt::xarray<bool> accept_mask;
  std::optional<xt::xarray<float>> mean_image;
  xt::xarray<float> selfnoise_offset;
  xt::xarray<float> selfnoise_raw;
  xt::xarray<float> dde_noise;
  std::optional<xt::xarray<float>> flux_cutoff;
  std::optional<xt::xarray<float>> snr_cutoff;
};

/**
 * @brief One already-loaded lot: every plane to process, plus the groups they reference — the input to
 *        run_lot_core().
 */
struct LotBatch
{
  std::vector<PlaneBatchItem> planes;
  std::vector<PlaneBatchGroup> groups; // indexed by PlaneBatchItem::group_index
};

/**
 * @brief Detection table columns, one entry per island found across every band of a plane — the C++ analog of
 *        breifast.py's `DetectionTable`.
 */
struct DetectionArrays
{
  std::vector<long> peak_x;
  std::vector<long> peak_y;
  std::vector<long> island_size;

  std::vector<std::string> stokes_label;
  std::vector<std::string> band_label;

  std::vector<float> local_noise;
  std::vector<float> local_noise_median_ratio;

  std::vector<float> signal_to_noise;
  std::vector<float> signal_to_noise_rms;
  std::vector<float> signal_to_noise_extended_box;
  std::vector<float> signal_to_noise_local_mad;
  std::vector<float> signal_to_noise_global_mad;

  std::vector<float> self_noise;
  std::vector<float> excess_noise_factor;
  std::vector<float> beam_chi2;
  std::vector<float> boost;

  void reserve(const size_t size)
  {
    peak_x.reserve(size);
    peak_y.reserve(size);
    island_size.reserve(size);
    stokes_label.reserve(size);
    band_label.reserve(size);
    local_noise.reserve(size);
    local_noise_median_ratio.reserve(size);
    signal_to_noise.reserve(size);
    signal_to_noise_rms.reserve(size);
    signal_to_noise_extended_box.reserve(size);
    signal_to_noise_local_mad.reserve(size);
    signal_to_noise_global_mad.reserve(size);
    self_noise.reserve(size);
    excess_noise_factor.reserve(size);
    beam_chi2.reserve(size);
    boost.reserve(size);
  }
};

/**
 * @brief Per-pixel spectral-variance QA diagnostics accumulated across a plane's frequency channels (not used for
 *        detection, only diagnostic aggregation further downstream in breifast.py).
 */
struct SpectralVarianceStats
{
  xt::xarray<float> sum_pow2;
  xt::xarray<float> sum_pow3;
  xt::xarray<float> sum_pow4;
  xt::xarray<bool> nonzero_channels_mask;
  std::optional<xt::xarray<float>> per_channels_squared;
};

/**
 * @brief Everything process_plane() produces for one plane: detections, sigma range, per-band histograms, and (if
 *        multi-channel) variance QA stats.
 */
struct ProcessPlaneResult
{
  bool any_data = false;
  DetectionArrays detections;
  std::optional<float> min_sigma;
  std::optional<float> max_sigma;
  std::vector<long> histogram0;
  std::vector<long> histogram1;
  std::vector<long> histogram2;
  std::vector<double> median_noises;
  std::optional<SpectralVarianceStats> variance_stats;
};

/**
 * @brief Local RMS noise estimate: a sliding `boxsize x boxsize` box filter over `plane`, edge-corrected so border
 *        windows use only their true (in-bounds) pixel count rather than being pulled down by phantom padding.
 *        Equivalent to `make_noise_map(..., method='std')` in breifast.py.
 *
 * Implemented via a summed-area table (2D prefix sum of `plane^2`, padded with a zero row/column at index 0) so
 * each pixel's windowed sum is four O(1) lookups instead of re-summing up to `boxsize^2` values per pixel.
 * @param plane 2D image to compute the noise map over.
 * @param boxsize Full window width/height (not radius).
 * @return Noise map, same shape as `plane`: `sqrt(mean of plane^2 within each pixel's window)`.
 */
xt::xarray<float> compute_rms_noise_map(const xt::xarray<float> &plane, long boxsize);

/**
 * @brief Convert `plane` to SNR units and apply the full per-pixel cutoff cascade in one pass
 *        (breifast.py:741-804): SNR/valid-noisemap masking, optional `snr_cutoff`/`flux_cutoff`/
 *        `reject_noise_threshold` filters, histogram accumulation, plane-level min/max sigma tracking, and finally
 *        the `threshold` mask.
 *
 * `image_mask` is refined in place through four cumulative (never-widening) stages per pixel;
 * `result.histogram0`/`1`/`2` are overwritten (not accumulated) each call, deliberately matching breifast.py:750's
 * per-band reset — the caller only ever sees the last band processed, not a sum across bands.
 * `result.min_sigma`/`max_sigma` persist and combine across every `process_band()` call for the plane.
 * @param plane Band's weighted (or raw, for single-channel bands) image.
 * @param noise_map Combined noise map for this band (see process_band()).
 * @param median_noise Median of `noise_map`, used by the `reject_noise_threshold` filter.
 * @param inv_band_size `1 / band.size()`, used to average `snr_cutoff`/`flux_cutoff` over the band's channel range.
 * @param image_mask In/out: narrowed in place to the final per-pixel mask.
 * @param group Supplies the optional `snr_cutoff`/`flux_cutoff` arrays.
 * @param band Supplies the channel range to average cutoff arrays over.
 * @param config Supplies `max_sigma_bin`, `histogram_detail`, `threshold`, `reject_noise_threshold`.
 * @param result Histogram and min/max-sigma outputs are written here.
 * @return SNR image, same shape as `plane` (needed later for peak/island work).
 */
xt::xarray<float> compute_snr_and_apply_filters(const xt::xarray<float> &plane, const xt::xarray<float> &noise_map,
                                                float median_noise, float inv_band_size, xt::xarray<bool> &image_mask,
                                                const PlaneBatchGroup &group, const BandSlice &band,
                                                const PlaneConfig &config, ProcessPlaneResult &result);

/**
 * @brief Build a centred, rotated, unit-sum-normalized 2D anisotropic Gaussian kernel.
 *
 * Specializes `cubes.py`'s `create_multidimensional_gaussian_kernel` for the `centred=True`, odd-square-size case
 * used for beam fitting: axis offsets run `-halfsize..halfsize`, are rotated CCW by `rotate` before evaluating the
 * Gaussian, and the result is normalized to sum to 1.
 * @param halfsize Kernel half-width; full size is `2*halfsize+1` on each axis.
 * @param sigma_major Gaussian sigma along the pre-rotation first axis.
 * @param sigma_minor Gaussian sigma along the pre-rotation second axis.
 * @param rotate Counter-clockwise rotation, in radians.
 * @return `(2*halfsize+1) x (2*halfsize+1)` kernel, normalized so its elements sum to 1.
 */
xt::xarray<float> build_gaussian_beam_kernel(long halfsize, float sigma_major, float sigma_minor, float rotate);

/**
 * @brief Process one band slice of one plane: mask, weight, denoise, convert to SNR, label islands, and compute
 *        every per-detection statistic for this band.
 *
 * Mirrors breifast.py's per-band-slice body inside the plane loop. Steps, each with early-exit checks so later,
 * more expensive work is skipped when possible: OR-reduce `accept_mask` across the band's channels; compute the
 * weighted-mean (or raw, for single-channel bands) image with optional mean subtraction; combine an RMS-box or
 * global-MAD noise estimate with the group's self-noise/DDE-noise maps; convert to SNR and apply cutoff filters via
 * compute_snr_and_apply_filters(). Label connected-component islands via breifast::islands::label_islands(), then
 * for each island extract a zero-padded postage stamp to estimate adjacent noise (feeding the extended-box/RMS/
 * local-MAD SNR variants), optionally fit a beam kernel for `beam_chi2`, and optionally compute flux `boost`
 * against `config.baseline_flux`. Writes histogram/min-max-sigma outputs and detections into `result`, shared
 * across every band of the plane.
 * @param item Plane data (stokes cube, weights, nonzero-channel mask, optional beam axes) to process. Mutated for
 *        single-channel bands, which overwrite their channel's `stokes_cube` slice with SNR values.
 * @param group Shared per-Stokes arrays (accept mask, mean image, noise maps).
 * @param band Channel range this call processes.
 * @param config Run-level configuration.
 * @param result Accumulator shared across all of this plane's process_band() calls.
 */
void process_band(PlaneBatchItem &item, const PlaneBatchGroup &group, const BandSlice &band, const PlaneConfig &config,
                  ProcessPlaneResult &result);

/**
 * @brief Compute per-pixel spectral-variance QA stats across all of a plane's frequency channels (raw sums of
 *        squares/cubes/4th powers per pixel).
 *
 * Diagnostic aggregation only, not part of source detection — see breifast.py:960-1095's
 * `spvar`/`spskew`/`spkurt` computation and its running-sum accumulation in `process_plane_result`
 * (breifast.py:1107-1153).
 *
 * @param item Plane data providing the channel cube to accumulate over.
 * @param group Shared per-Stokes arrays.
 * @param config Run-level configuration.
 * @return Accumulated per-pixel moment sums and channel-validity mask.
 */
SpectralVarianceStats compute_spectral_variance_stats(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                                      const PlaneConfig &config);

/**
 * @brief Process every band of one plane, then (for multi-channel planes) compute plane-wide spectral-variance
 *        QA stats.
 * @param config Run-level configuration, including the band slices to process.
 * @param item Plane data to process.
 * @param group Shared per-Stokes arrays `item` references.
 * @return Combined detections/histograms/sigma-range/variance-stats for the plane.
 */
ProcessPlaneResult process_plane(const PlaneConfig &config, PlaneBatchItem &item, const PlaneBatchGroup &group);

/**
 * @brief Run process_plane() over every item in `items`, in parallel via TBB.
 *
 * Each item's group is looked up by `group_index` into `groups`. Per-item exceptions are caught and logged rather
 * than aborting the whole batch, so one bad plane doesn't take down the rest.
 * @param items Planes to process; not modified, but non-const to match TBB's expected iteration interface.
 * @param groups Shared per-Stokes arrays, indexed by `PlaneBatchItem::group_index`.
 * @param config Run-level configuration shared by every item.
 * @param num_threads TBB arena size; `<= 0` means `tbb::task_arena::automatic`.
 * @return One `ProcessPlaneResult` per item, in `items` order.
 */
std::vector<ProcessPlaneResult> run_batch_core(std::vector<PlaneBatchItem> &items,
                                               const std::vector<PlaneBatchGroup> &groups, const PlaneConfig &config,
                                               int num_threads);

/**
 * @brief Replaces breifast.py:1173-1192's futures-submit-and-wait loop for one already-loaded lot.
 *
 * Thin wrapper over run_batch_core(). Caller (Python) still does the per-plane breifast.py:1192
 * `process_plane_result(*result)` unpacking, same as today.
 * @param lot Planes and groups to process.
 * @param config Run-level configuration.
 * @param num_threads TBB arena size; `0` (default) means automatic.
 * @return One `ProcessPlaneResult` per plane, in `lot.planes` order.
 */
inline std::vector<ProcessPlaneResult> run_lot_core(LotBatch &lot, const PlaneConfig &config, int num_threads = 0)
{
  return run_batch_core(lot.planes, lot.groups, config, num_threads);
}
} // namespace breifast