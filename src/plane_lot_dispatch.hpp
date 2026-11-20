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
 */
#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xshape.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/reducers/xreducer.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>

#include "cpu/beam_kernel.hpp"
#include "cpu/islands.hpp"
#include "detection_arrays.hpp"
#include "host_memory.hpp"
#include "process_plane_result.hpp"

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
          return std::abs(x_prev + (x_val - x_prev) * (0.5f - F_prev) / (F_val - F_prev));
        }
        x_prev = x_val;
        f_prev = f_val;
        F_prev = F_val;
      }
      return std::abs(x_prev);
    };

    static std::unordered_map<size_t, float> cache;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock{cache_mutex};
    const auto it = cache.find(n);
    if (it != cache.cend())
    {
      return it->second;
    }
    const auto value = compute(n);
    cache.emplace(n, value);
    return value;
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
  std::optional<xt::xtensor<double, 2>> fixed_beam_kernel;
  long fixed_beam_halfsize = 0;
  std::optional<xt::xtensor<float, 2>> baseline_flux;

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
  HostCube stokes_cube; // pooled, page-locked in GPU builds: the GPU uploads it at full speed
  xt::xtensor<float, 1> weights;
  xt::xtensor<bool, 1> nonzero_channels;
  std::string stokes_label;
  std::optional<xt::xtensor<float, 1>> beam_major_axis, beam_minor_axis, beam_position_angle;
};

/**
 * @brief Per-Stokes-label arrays shared by every `PlaneBatchItem` that references them via `group_index` (accept
 *        mask, mean image, noise/cutoff maps).
 */
struct PlaneBatchGroup
{
  xt::xtensor<bool, 3> accept_mask;
  std::optional<xt::xtensor<float, 3>> mean_image;
  xt::xtensor<float, 2> selfnoise_offset;
  xt::xtensor<float, 2> selfnoise_raw;
  xt::xtensor<float, 2> dde_noise;
  std::optional<xt::xtensor<float, 3>> flux_cutoff;
  std::optional<xt::xtensor<float, 3>> snr_cutoff;
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

/// The beam of one band: its half size, and its kernel, empty when the beam is disabled.
struct BeamKernel
{
  size_t halfsize = 0;
  std::optional<xt::xtensor<float, 2>> kernel;
};

/// Half size and kernel of the beam for the band: the fixed kernel of the config, or a Gaussian built from the band's
/// weighted beam axes. Both backends call it, so they use the same kernel.
inline BeamKernel beam_kernel_for_band(const PlaneBatchItem &item, const BandSlice &band, const PlaneConfig &config)
{
  BeamKernel beam;
  if (config.beam_size_enabled)
  {
    if (config.fixed_beam_kernel.has_value())
    {
      beam.halfsize = config.fixed_beam_halfsize;
      beam.kernel = xt::xtensor<float, 2>(*config.fixed_beam_kernel);
    }
    else
    {
      float beam_kernel_halfsize_flt = 0.F;
      for (auto i_channel = band.start; i_channel < band.stop; ++i_channel)
      {
        beam_kernel_halfsize_flt += (*item.beam_major_axis)(i_channel);
      }
      beam_kernel_halfsize_flt *= 3 / static_cast<float>(band.size());
      beam.halfsize = static_cast<long>(beam_kernel_halfsize_flt) + 1;

      // The band weights and their sum, computed as process_band computes them.
      const xt::xtensor<float, 1> weights_sliced = xt::view(item.weights, band.xrange());
      const float weights_sum = std::accumulate(weights_sliced.cbegin(), weights_sliced.cend(), 0.F);
      const auto beam_major_sliced = xt::view(*item.beam_major_axis, band.xrange());
      const auto beam_minor_sliced = xt::view(*item.beam_minor_axis, band.xrange());
      const auto beam_position_angle_sliced = xt::view(*item.beam_position_angle, band.xrange());
      const float beam_major_weighted = xt::sum(beam_major_sliced * weights_sliced)() / weights_sum;
      const float beam_minor_weighted = xt::sum(beam_minor_sliced * weights_sliced)() / weights_sum;
      const auto beam_position_angle_weighted = xt::sum(beam_position_angle_sliced * weights_sliced)() / weights_sum;
      beam.kernel = build_gaussian_beam_kernel(beam.halfsize, beam_major_weighted, beam_minor_weighted,
                                               beam_position_angle_weighted);
    }
  }
  return beam;
}

struct ProcessBandScratch
{
  xt::xtensor<float, 2> row_window_sums;
  std::vector<double> column_accumulators;
  xt::xtensor<float, 2> snr;
  std::array<std::vector<long>, 3> histograms;
  xt::xtensor<float, 2> windows_sums;
  xt::xtensor<float, 2> plane;
  std::vector<float> median_scratch;
  xt::xtensor<bool, 2> image_mask;
  xt::xtensor<float, 2> noise_map;
  xt::xtensor<float, 2> stamp;
  xt::xtensor<float, 2> beam_stamp;
  islands::IslandLabels islands;
};

/**
 * @brief Local RMS noise estimate: a sliding `boxsize x boxsize` box filter over `plane`, edge-corrected so border
 *        windows use only their true (in-bounds) pixel count.
 *
 * Two separable passes: running sums of `plane^2` along each row into `scratch.row_window_sums`, then along each
 * column into `scratch.windows_sums`. The caller applies `make_noise_map(method='std')`'s median floor.
 * @param plane 2D image to compute the noise map over.
 * @param boxsize Full window width/height.
 * @return `sqrt(mean of plane^2 within each pixel's window)`, same shape as `plane`. Refers to
 *         `scratch.windows_sums`, so it stays valid only until the next call sharing that scratch.
 */
const xt::xtensor<float, 2> &compute_rms_noise_map(ProcessBandScratch &scratch, const xt::xtensor<float, 2> &plane,
                                                   long boxsize);

/**
 * @brief Convert `plane` to SNR units and apply the full per-pixel cutoff cascade in one pass
 *        (breifast.py:741-804): SNR/valid-noisemap masking, optional `snr_cutoff`/`flux_cutoff`/
 *        `reject_noise_threshold` filters, histogram accumulation, plane-level min/max sigma tracking, and finally
 *        the `threshold` mask.
 *
 * `image_mask` is refined in place through four cumulative (never-widening) stages per pixel. `scratch.snr` and
 * `scratch.histograms` are resized-and-reused scratch buffers; `result.histograms` is overwritten from
 * `scratch.histograms` each call.
 * `result.min_sigma`/`max_sigma` persist and combine across every `process_band()` call for the plane.
 * @param scratch Supplies `scratch.snr` (resized to `plane`'s shape) and `scratch.histograms` (resized to
 *        `config.max_sigma_bin * 2`), both reused across calls to avoid a per-call allocation.
 * @param plane Band's weighted (or raw, for single-channel bands) image.
 * @param noise_map Combined noise map for this band (see process_band()).
 * @param median_noise Median of `noise_map`, used by the `reject_noise_threshold` filter.
 * @param inv_band_size `1 / band.size()`, used to average `snr_cutoff`/`flux_cutoff` over the band's channel range.
 * @param image_mask In/out: narrowed in place to the final per-pixel mask.
 * @param group Supplies the optional `snr_cutoff`/`flux_cutoff` arrays.
 * @param band Supplies the channel range to average cutoff arrays over.
 * @param config Supplies `max_sigma_bin`, `histogram_detail`, `threshold`, `reject_noise_threshold`.
 * @param result Histogram and min/max-sigma outputs are written here.
 */
void compute_snr_and_apply_filters(ProcessBandScratch &scratch, const xt::xtensor<float, 2> &plane,
                                   const xt::xtensor<float, 2> &noise_map, float median_noise, float inv_band_size,
                                   xt::xtensor<bool, 2> &image_mask, const PlaneBatchGroup &group,
                                   const BandSlice &band, const PlaneConfig &config, ProcessPlaneResult &result);

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
void process_band(ProcessBandScratch &scratch, PlaneBatchItem &item, const PlaneBatchGroup &group,
                  const BandSlice &band, const PlaneConfig &config, ProcessPlaneResult &result);

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
ProcessPlaneResult process_plane(ProcessBandScratch &scratch, const PlaneConfig &config, PlaneBatchItem &item,
                                 const PlaneBatchGroup &group);

/**
 * @brief Throws std::invalid_argument, naming the plane and the field, when a plane is inconsistent with itself, its
 *        group or the config: array shapes, band ranges, group index, beam inputs and config values.
 *
 * Both backends run it on every plane before any work, so a bad input fails the whole call up front instead of one
 * plane midway.
 */
void validate_plane_inputs(const std::vector<PlaneBatchItem> &items, const std::vector<PlaneBatchGroup> &groups,
                           const PlaneConfig &config);

/**
 * @brief Run process_plane() over every item in `items`, in parallel via TBB.
 *
 * Each item's group is looked up by `group_index` into `groups`. Every plane is validated first
 * (validate_plane_inputs()). If a plane throws during the run, the planes not started yet are skipped and its
 * exception is rethrown once the workers have stopped.
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