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
#include <exception>
#include <limits>
#include <numeric>
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

#include "islands.hpp"
#include "xtensor/core/xoperation.hpp"

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
  static float max_to_std_ratio(const size_t n)
  {
    const auto compute = [](const size_t n)
    {
      constexpr size_t n_samples = 1'000;
      constexpr float x_ampl = 10.F;
      float x_prev = -10.F;
      float f_prev = 0.5F * (1.F + std::erf(x_prev / std::sqrt(2.F)));
      float F_prev = 1.F - std::pow(1.f - f_prev, n);
      for (size_t i = 1; i < n_samples; ++i)
      {
        const auto x_val = -x_ampl + static_cast<float>(i) * 2.F * x_ampl / static_cast<float>(n_samples - 1);
        const auto f_val = 0.5F * (1.F + std::erf(x_val / std::sqrt(2.F)));
        const auto F_val = 1.F - std::pow(1.f - f_val, static_cast<float>(n));
        if (F_val >= 0.5F)
        {
          return x_prev + (x_val - x_prev) * (0.5f - F_prev) / (F_val - F_prev);
        }
        x_prev = x_val;
        f_prev = f_val;
        F_prev = F_val;
      }
      return x_prev;
    };

    static tbb::concurrent_unordered_map<size_t, float> cache;
    static tbb::mutex compute_mutex;
    const auto it = cache.find(n);
    if (it != cache.cend())
    {
      return it->second;
    }
    else
    {
      tbb::mutex::scoped_lock lock{compute_mutex};
      const auto value = compute(n);
      cache.emplace(n, value);
      return value;
    }
  }

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
  float max_to_std_ratio_value() const
  {
    if (max_to_std_ratio_value_.has_value())
    {
      return *max_to_std_ratio_value_;
    }
    const auto radius = boxsize / 2;
    const auto n = (2 * radius + 1) * (2 * radius + 1);
    max_to_std_ratio_value_ = max_to_std_ratio(n);
    return *max_to_std_ratio_value_;
  }
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
inline xt::xarray<float> compute_rms_noise_map(const xt::xarray<float> &plane, long boxsize)
{
  const long n_y = static_cast<long>(plane.shape(0));
  const long n_x = static_cast<long>(plane.shape(1));

  xt::xarray<float> parts_sums;
  parts_sums.reshape(xt::shape({static_cast<size_t>(n_y + 1), static_cast<size_t>(n_x + 1)}));
  for (long i_y = 0; i_y <= n_y; ++i_y)
  {
    parts_sums(i_y, 0) = 0.F;
  }
  for (long i_x = 0; i_x <= n_x; ++i_x)
  {
    parts_sums(0, i_x) = 0.F;
  }
  for (long i_y = 0; i_y < n_y; ++i_y)
  {
    for (long i_x = 0; i_x < n_x; ++i_x)
    {
      parts_sums(i_y + 1, i_x + 1) = plane(i_y, i_x) * plane(i_y, i_x) + parts_sums(i_y, i_x + 1) +
                                     parts_sums(i_y + 1, i_x) - parts_sums(i_y, i_x);
    }
  }

  xt::xarray<float> windows_sums;
  windows_sums.reshape(xt::shape({static_cast<size_t>(n_y), static_cast<size_t>(n_x)}));

  for (long i_y = 0; i_y < n_y; ++i_y)
  {
    for (long i_x = 0; i_x < n_x; ++i_x)
    {
      const long min_x = std::max(i_x - boxsize / 2, 0L);
      const long max_x = std::min(i_x + boxsize / 2 + 1, n_x);
      const long min_y = std::max(i_y - boxsize / 2, 0L);
      const long max_y = std::min(i_y + boxsize / 2 + 1, n_y);

      const float window_sum =
          parts_sums(max_y, max_x) - parts_sums(min_y, max_x) - parts_sums(max_y, min_x) + parts_sums(min_y, min_x);
      const float count = static_cast<float>(max_y - min_y) * static_cast<float>(max_x - min_x);

      windows_sums(i_y, i_x) = std::sqrt(window_sum / count);
    }
  }

  return windows_sums;
}

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
inline xt::xarray<float> compute_snr_and_apply_filters(const xt::xarray<float> &plane,
                                                       const xt::xarray<float> &noise_map, float median_noise,
                                                       float inv_band_size, xt::xarray<bool> &image_mask,
                                                       const PlaneBatchGroup &group, const BandSlice &band,
                                                       const PlaneConfig &config, ProcessPlaneResult &result)
{
  const size_t n_y = plane.shape(0);
  const size_t n_x = plane.shape(1);

  xt::xarray<float> snr;
  snr.reshape(plane.shape());
  result.histogram0.resize(config.max_sigma_bin * 2);
  result.histogram1.resize(config.max_sigma_bin * 2);
  result.histogram2.resize(config.max_sigma_bin * 2);
  float local_min_sigma = std::numeric_limits<float>::infinity();
  float local_max_sigma = std::numeric_limits<float>::lowest();
  for (size_t i_y = 0; i_y < n_y; ++i_y)
  {
    for (size_t i_x = 0; i_x < n_x; ++i_x)
    {
      const auto noise = noise_map(i_y, i_x);
      const bool valid = noise != 0 && std::isfinite(noise);
      snr(i_y, i_x) = valid ? plane(i_y, i_x) / noise_map(i_y, i_x) : 0;
      image_mask(i_y, i_x) &= valid;

      const auto mask0 = image_mask(i_y, i_x);

      float snr_cutoff_threshold = 0.F;
      const bool snr_cutoff_enabled = group.snr_cutoff.has_value();
      if (snr_cutoff_enabled)
      {
        for (auto i_channel = band.start; i_channel < band.stop; ++i_channel)
        {
          snr_cutoff_threshold += (*group.snr_cutoff)(i_channel, i_y, i_x);
        }
        snr_cutoff_threshold *= inv_band_size;
      }

      float flux_cutoff_threshold = 0.F;
      const bool flux_cutoff_enabled = group.flux_cutoff.has_value();
      if (flux_cutoff_enabled)
      {
        for (auto i_channel = band.start; i_channel < band.stop; ++i_channel)
        {
          flux_cutoff_threshold += (*group.flux_cutoff)(i_channel, i_y, i_x);
        }
        flux_cutoff_threshold *= inv_band_size;
      }

      const auto mask1 = mask0 && (!snr_cutoff_enabled || std::abs(snr(i_y, i_x)) >= snr_cutoff_threshold);
      const auto mask2 = mask1 && (!flux_cutoff_enabled || std::abs(plane(i_y, i_x)) >= flux_cutoff_threshold);
      const auto mask3 = mask2 && (config.reject_noise_threshold == 0.F ||
                                   noise_map(i_y, i_x) <= median_noise * config.reject_noise_threshold);

      image_mask(i_y, i_x) = mask3;

      long bin_index = std::floor(snr(i_y, i_x) + config.max_sigma_bin);
      if (bin_index == 2 * config.max_sigma_bin)
      {
        bin_index = 2 * config.max_sigma_bin - 1;
      }
      if (0 <= bin_index && bin_index < 2 * config.max_sigma_bin)
      {
        if (mask0 && config.histogram_detail > 1)
        {
          result.histogram0[bin_index]++;
        }
        if (mask1 && config.histogram_detail > 1)
        {
          result.histogram1[bin_index]++;
        }
        if (mask2 && config.histogram_detail > 0)
        {
          result.histogram2[bin_index]++;
        }
      }

      if (mask3)
      {
        const float value = snr(i_y, i_x);
        local_min_sigma = std::min(local_min_sigma, value);
        local_max_sigma = std::max(local_max_sigma, value);
      }

      const float thresholded_abs_snr = mask3 ? std::abs(snr(i_y, i_x)) : 0.F;
      image_mask(i_y, i_x) = thresholded_abs_snr >= config.threshold;
    }
  }
  if (local_min_sigma < std::numeric_limits<float>::infinity())
  {
    result.min_sigma = result.min_sigma ? std::min(*result.min_sigma, local_min_sigma) : local_min_sigma;
    result.max_sigma = result.max_sigma ? std::max(*result.max_sigma, local_max_sigma) : local_max_sigma;
  }
  return snr;
}

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
xt::xarray<float> build_gaussian_beam_kernel(long halfsize, float sigma_major, float sigma_minor, float rotate)
{
  const auto size = 2 * halfsize + 1;
  xt::xarray<float> kernel;
  kernel.reshape({static_cast<size_t>(size), static_cast<size_t>(size)});

  const float cos_r = std::cos(rotate);
  const float sin_r = std::sin(rotate);
  float sum = 0.F;
  for (long i = 0; i < size; ++i)
  {
    const float dy = static_cast<float>(i - halfsize);
    for (long j = 0; j < size; ++j)
    {
      const float dx = static_cast<float>(j - halfsize);
      const float rot_dy = cos_r * dy - sin_r * dx;
      const float rot_dx = sin_r * dy + cos_r * dx;
      const float value = std::exp(
          -(rot_dy * rot_dy / (2.F * sigma_major * sigma_major) + rot_dx * rot_dx / (2.F * sigma_minor * sigma_minor)));
      kernel(i, j) = value;
      sum += value;
    }
  }
  kernel /= sum;
  return kernel;
}

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
inline void process_band(PlaneBatchItem &item, const PlaneBatchGroup &group, const BandSlice &band,
                         const PlaneConfig &config, ProcessPlaneResult &result)
{
  if (!xt::any(xt::view(item.nonzero_channels, band.xrange())))
  {
    return;
  }

  const size_t n_y = item.stokes_cube.shape(1);
  const size_t n_x = item.stokes_cube.shape(2);

  // Masking
#if USE_XTENSOR
  const auto band_accept_mask = xt::view(group.accept_mask, band.xrange(), xt::all(), xt::all());
  const xt::xarray<bool> image_mask = xt::reduce(std::logical_or<bool>(), band_accept_mask, {0});
  if (!xt::any(image_mask))
  {
    return;
  }
#else
  xt::xarray<bool> image_mask = xt::view(group.accept_mask, band.start, xt::all(), xt::all());
  bool any_mask = xt::any(image_mask);
  for (long i_band = band.start + 1; i_band < band.stop; ++i_band)
  {
    for (size_t i_y = 0; i_y < n_y; ++i_y)
    {
      for (size_t i_x = 0; i_x < n_x; ++i_x)
      {
        const bool mask = group.accept_mask(i_band, i_y, i_x) || image_mask(i_y, i_x);
        any_mask |= mask;
        image_mask(i_y, i_x) = mask;
      }
    }
  }

  if (!any_mask)
  {
    return;
  }
#endif

  const auto &weights = item.weights;
  const xt::xarray<float> weights_sliced = xt::view(weights, band.xrange());

  const bool do_subtract_mean = config.subtract_mean && group.mean_image.has_value();
  const float inv_band_size = 1.F / static_cast<float>(band.size());
#if USE_XTENSOR
  xt::xarray<float> plane = xt::view(item.stokes_cube, band.xrange(), xt::all(), xt::all());
  const auto weights_sum = xt::sum(weights_sliced);
  if (band.size() > 1)
  {
    plane *= xt::view(weights_sliced, xt::all(), xt::newaxis(), xt::newaxis());
    plane = xt::sum(plane, {0});
    plane *= (1.F / weights_sum);
  }

  if (do_subtract_mean)
  {
    const xt::xarray<float> sliced_mean = xt::view(*group.mean_image, band.xrange(), xt::all(), xt::all());
    plane = plane - xt::sum(sliced_mean, {0}) * inv_band_size;
  }
#else
  const float weights_sum = std::accumulate(weights_sliced.cbegin(), weights_sliced.cend(), 0.F);
  const float inv_weights_sum = 1.F / weights_sum;
  xt::xarray<float> plane;
  plane.reshape(xt::shape({n_y, n_x}));
  plane.fill(0);
  for (auto i_channel = band.start; i_channel < band.stop; ++i_channel)
  {
    for (size_t i_y = 0; i_y < n_y; ++i_y)
    {
      for (size_t i_x = 0; i_x < n_x; ++i_x)
      {
        plane(i_y, i_x) += item.stokes_cube(i_channel, i_y, i_x) * weights(i_channel) * inv_weights_sum;
        if (do_subtract_mean)
        {
          plane(i_y, i_x) -= (*group.mean_image)(i_channel, i_y, i_x) * inv_band_size;
        }
      }
    }
  }
#endif

  const float noise_global_mad = xt::median(xt::abs(plane)) * 1.4826F;

  const auto combine = [&](const auto &noisemap0)
  {
    xt::xarray<float> noisemap = xt::sqrt(noisemap0 * noisemap0 + band.selfnoise_scale_factor * group.selfnoise_offset +
                                          config.ddenoise_scaling_factor * group.dde_noise);
    xt::xarray<float> excess_noise_factor = noisemap / noisemap0;
    const float mednoise = xt::median(noisemap);
    return std::tuple{noisemap, excess_noise_factor, mednoise};
  };

  xt::xarray<float> noise_map;
  xt::xarray<float> excess_noise_factor;
  float median_noise = 0.F;
  if (config.use_rms_noise_estimator)
  {
    std::tie(noise_map, excess_noise_factor, median_noise) = combine(compute_rms_noise_map(plane, config.boxsize));
  }
  else
  {
    std::tie(noise_map, excess_noise_factor, median_noise) = combine(noise_global_mad);
  }

  const auto snr = compute_snr_and_apply_filters(plane, noise_map, median_noise, inv_band_size, image_mask, group, band,
                                                 config, result);
  if (band.size() == 1)
  {
    xt::view(item.stokes_cube, band.start) = snr;
  }
  if (!xt::any(image_mask))
  {
    return;
  }

  const auto radius = config.boxsize / 2;
  long pad;
  long beam_kernel_halfsize = 0;
  if (config.beam_size_enabled)
  {
    if (config.fixed_beam_kernel.has_value())
    {
      beam_kernel_halfsize = config.fixed_beam_halfsize;
    }
    else
    {
      float beam_kernel_halfsize_flt = 0.F;
      for (auto i_channel = band.start; i_channel < band.stop; ++i_channel)
      {
        beam_kernel_halfsize_flt += (*item.beam_major_axis)(i_channel);
      }
      beam_kernel_halfsize_flt *= 3 / static_cast<float>(band.size());
      beam_kernel_halfsize = static_cast<long>(beam_kernel_halfsize_flt) + 1;
    }
    pad = std::max(beam_kernel_halfsize, radius + 1);
  }
  else
  {
    pad = radius + 1;
  }

  xt::xarray<float> padded_image;
  padded_image.reshape({plane.shape(0) + pad * 2, plane.shape(1) + pad * 2});

  for (size_t i_y = 0; i_y < padded_image.shape(0); ++i_y)
  {
    for (size_t i_x = 0; i_x < padded_image.shape(1); ++i_x)
    {
      const bool in_padding = i_y < pad || i_y < pad || i_y > pad + n_y || i_x > pad + n_x;
      padded_image(i_y, i_x) = in_padding ? 0 : plane(i_y - pad, i_x - pad);
    }
  }

  const auto islands = breifast::islands::label_islands(image_mask, snr, noise_map);

  std::optional<xt::xarray<float>> beam_kernel;
  if (config.beam_size_enabled)
  {
    if (config.fixed_beam_kernel.has_value())
    {
      beam_kernel = xt::xarray<float>(*config.fixed_beam_kernel);
    }
    else
    {
      const auto beam_major_sliced = xt::view(*item.beam_major_axis, band.xrange());
      const auto beam_minor_sliced = xt::view(*item.beam_minor_axis, band.xrange());
      const auto beam_position_angle_sliced = xt::view(*item.beam_position_angle, band.xrange());
      const float beam_major_weighted = xt::sum(beam_major_sliced * weights_sliced)() / weights_sum;
      const float beam_minor_weighted = xt::sum(beam_minor_sliced * weights_sliced)() / weights_sum;
      const auto beam_position_angle_weighted = xt::sum(beam_position_angle_sliced * weights_sliced)() / weights_sum;
      beam_kernel = build_gaussian_beam_kernel(beam_kernel_halfsize, beam_major_weighted, beam_minor_weighted,
                                               beam_position_angle_weighted);
    }
  }

  DetectionArrays detection_arrays;
  detection_arrays.reserve(islands.peak_per_label.size());
  for (const auto [label, island] : islands.peak_per_label)
  {
    const auto padded_y = island.coordinates.first + pad;
    const auto padded_x = island.coordinates.second + pad;

    xt::xarray<float> sub_image = xt::view(padded_image, xt::range(padded_y - radius, padded_y + radius + 1),
                                           xt::range(padded_x - radius, padded_x + radius + 1));
    xt::view(sub_image, xt::range(radius - config.inner_boxsize, radius + config.inner_boxsize + 1),
             xt::range(radius - config.inner_boxsize, radius + config.inner_boxsize + 1)) = 0.F;

    const auto minmax_subimage = xt::minmax(sub_image)();
    const auto extreme = std::max(std::abs(minmax_subimage[0]), std::abs(minmax_subimage[1]));
    const auto adjacent_noise = extreme / config.max_to_std_ratio_value();

    const auto flux = plane(island.coordinates.first, island.coordinates.second);
    const auto signal_to_noise_extended_box = adjacent_noise != 0.F ? flux / adjacent_noise : 0.F;

    float beam_chi2_value = 0.F;
    if (beam_kernel.has_value())
    {
      xt::xarray<float> beam_sub_image =
          xt::view(padded_image, xt::range(padded_y - beam_kernel_halfsize, padded_y + beam_kernel_halfsize + 1),
                   xt::range(padded_x - beam_kernel_halfsize, padded_x + beam_kernel_halfsize + 1));

      const auto fit = *beam_kernel * (xt::sum(beam_sub_image)() / xt::sum(*beam_kernel)());
      const auto mask = xt::not_equal(beam_sub_image, 0.F);
      const auto diff = xt::filter(beam_sub_image, mask) - xt::filter(fit, mask);
      const auto mask_count = xt::sum(mask)();

      if (mask_count > 0 && adjacent_noise != 0.F)
      {
        beam_chi2_value = xt::sum(diff * diff)() / (adjacent_noise * adjacent_noise * mask_count);
      }
    }

    const auto donut = xt::filter(sub_image, xt::not_equal(sub_image, 0.F));
    const auto rms = xt::stddev(donut)();
    const auto lmad = xt::median(xt::abs(donut)) * 1.4826F;

    float boost_value = 0.F;
    if (config.baseline_flux.has_value())
    {
      const auto island_mask = xt::equal(islands.labels, label);
      const xt::xarray<float> ratio = flux / xt::filter(*config.baseline_flux, island_mask);
      const auto max_abs_index = xt::argmax(xt::abs(ratio))();
      boost_value = ratio(max_abs_index);
    }

    detection_arrays.peak_y.push_back(island.coordinates.first);
    detection_arrays.peak_x.push_back(island.coordinates.second);
    detection_arrays.island_size.push_back(island.island_size);
    detection_arrays.stokes_label.push_back(item.stokes_label);
    detection_arrays.band_label.push_back(band.label);
    detection_arrays.local_noise.push_back(island.local_noise);
    detection_arrays.local_noise_median_ratio.push_back(island.local_noise / median_noise);
    detection_arrays.signal_to_noise.push_back(island.value);
    detection_arrays.signal_to_noise_rms.push_back(rms != 0.F ? flux / rms : 0.F);
    detection_arrays.signal_to_noise_extended_box.push_back(signal_to_noise_extended_box);
    detection_arrays.signal_to_noise_local_mad.push_back(lmad != 0.F ? flux / lmad : 0.F);
    detection_arrays.signal_to_noise_global_mad.push_back(flux / noise_global_mad);
    detection_arrays.self_noise.push_back(group.selfnoise_raw(island.coordinates.first, island.coordinates.second));
    detection_arrays.excess_noise_factor.push_back(
        excess_noise_factor(island.coordinates.first, island.coordinates.second));
    detection_arrays.beam_chi2.push_back(beam_chi2_value);
    detection_arrays.boost.push_back(boost_value);
  }
  result.detections = std::move(detection_arrays);
}

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
inline SpectralVarianceStats compute_spectral_variance_stats(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                                             const PlaneConfig &config)
{
  SpectralVarianceStats stats;
  const auto n_channels = item.stokes_cube.shape(0);
  const auto n_y = item.stokes_cube.shape(1);
  const auto n_x = item.stokes_cube.shape(2);

  stats.sum_pow2.reshape({n_y, n_x});
  stats.sum_pow3.reshape({n_y, n_x});
  stats.sum_pow4.reshape({n_y, n_x});

  stats.sum_pow2.fill(0.F);
  stats.sum_pow3.fill(0.F);
  stats.sum_pow4.fill(0.F);

  for (size_t i_channel = 0; i_channel < n_channels; ++i_channel)
  {
    if (!item.nonzero_channels(i_channel))
    {
      continue;
    }
    for (size_t i_y = 0; i_y < n_y; ++i_y)
    {
      for (size_t i_x = 0; i_x < n_x; ++i_x)
      {
        const auto value = item.stokes_cube(i_channel, i_y, i_x);
        const auto value2 = value * value;
        const auto value3 = value2 * value;
        const auto value4 = value3 * value;
        stats.sum_pow2(i_y, i_x) += value2;
        stats.sum_pow3(i_y, i_x) += value3;
        stats.sum_pow4(i_y, i_x) += value4;
      }
    }
  }

  stats.nonzero_channels_mask = item.nonzero_channels;

  return stats;
}

/**
 * @brief Process every band of one plane, then (for multi-channel planes) compute plane-wide spectral-variance
 *        QA stats.
 * @param config Run-level configuration, including the band slices to process.
 * @param item Plane data to process.
 * @param group Shared per-Stokes arrays `item` references.
 * @return Combined detections/histograms/sigma-range/variance-stats for the plane.
 */
inline ProcessPlaneResult process_plane(const PlaneConfig &config, PlaneBatchItem &item, const PlaneBatchGroup &group)
{
  ProcessPlaneResult result;
  for (const auto &slice : config.band_slices)
  {
    process_band(item, group, slice, config, result);
  }
  if (item.nonzero_channels.size() > 1)
  {
    result.variance_stats = compute_spectral_variance_stats(item, group, config);
  }

  return result;
}

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
inline std::vector<ProcessPlaneResult> run_batch_core(std::vector<PlaneBatchItem> &items,
                                                      const std::vector<PlaneBatchGroup> &groups,
                                                      const PlaneConfig &config, int num_threads)
{
  std::vector<ProcessPlaneResult> results(items.size());

  tbb::task_arena task_arena(num_threads > 0 ? num_threads : tbb::task_arena::automatic);

  const auto main_iteration = [&](const tbb::blocked_range<size_t> &range)
  {
    for (auto index = range.begin(); index != range.end(); ++index)
    {
      auto &item = items[index];
      const auto group_index = item.group_index;
      const auto &group = groups[group_index];

      try
      {
        results[index] = process_plane(config, item, group);
      }
      catch (const std::exception &e)
      {
        std::cerr << e.what() << std::endl;
      }
    }
  };
  const auto main_execution = [&]() { tbb::parallel_for(tbb::blocked_range<size_t>(0, items.size()), main_iteration); };

  task_arena.execute(main_execution);

  return results;
}

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
