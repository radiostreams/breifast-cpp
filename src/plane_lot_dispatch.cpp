#include <algorithm>
#include <span>
#include <type_traits>

#include <ittnotify.h>

#include <tracy/Tracy.hpp>

#include <xsimd/xsimd.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/reducers/xreducer.hpp>

#include "cpu/island_stamps.hpp"
#include "cpu/islands.hpp"
#include "cpu/order_statistics.hpp"
#include "cpu/pointwise.hpp"
#include "cpu/snr_filters.hpp"
#include "cpu/windows.hpp"
#include "plane_lot_dispatch.hpp"
#include "cpu/channel_reductions.hpp"
#include "cpu/scalar_reductions.hpp"

namespace breifast
{
float PlaneConfig::max_to_std_ratio(const size_t n)
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

float PlaneConfig::max_to_std_ratio_value() const
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

const xt::xtensor<float, 2> &compute_rms_noise_map(ProcessBandScratch &scratch, const xt::xtensor<float, 2> &plane,
                                                   long boxsize)
{
  const size_t n_y = plane.shape(0);
  const size_t n_x = plane.shape(1);
  scratch.windows_sums.resize({n_y, n_x});
  scratch.row_window_sums.resize({n_y, n_x});
  scratch.column_accumulators.resize(n_x);
  plane_lot::detail::rms_noise_map({plane.data(), plane.size()}, n_y, n_x, boxsize,
                                   {scratch.row_window_sums.data(), scratch.row_window_sums.size()},
                                   scratch.column_accumulators,
                                   {scratch.windows_sums.data(), scratch.windows_sums.size()});
  return scratch.windows_sums;
}

void compute_snr_and_apply_filters(ProcessBandScratch &scratch, const xt::xtensor<float, 2> &plane,
                                   const xt::xtensor<float, 2> &noise_map, float median_noise, float inv_band_size,
                                   xt::xtensor<bool, 2> &image_mask, const PlaneBatchGroup &group,
                                   const BandSlice &band, const PlaneConfig &config, ProcessPlaneResult &result)
{
  scratch.snr.resize(plane.shape());
  for (auto &histogram : scratch.histograms)
  {
    histogram.resize(config.max_sigma_bin * 2);
  }
  const auto cutoff_span = [](const std::optional<xt::xtensor<float, 3>> &cutoff)
  { return cutoff ? std::span<const float>(cutoff->data(), cutoff->size()) : std::span<const float>(); };

  const plane_lot::detail::SnrFilterParams params{median_noise,
                                                  inv_band_size,
                                                  band.start,
                                                  band.stop,
                                                  config.max_sigma_bin,
                                                  config.histogram_detail,
                                                  config.reject_noise_threshold,
                                                  config.threshold};
  const auto summary = plane_lot::detail::snr_and_filters(
      {plane.data(), plane.size()}, {noise_map.data(), noise_map.size()}, cutoff_span(group.snr_cutoff),
      cutoff_span(group.flux_cutoff), params, {scratch.snr.data(), scratch.snr.size()},
      {image_mask.data(), image_mask.size()},
      {std::span<long>(scratch.histograms[0]), std::span<long>(scratch.histograms[1]),
       std::span<long>(scratch.histograms[2])});

  result.histograms = scratch.histograms;
  if (summary.min_sigma < std::numeric_limits<float>::infinity())
  {
    result.min_sigma = result.min_sigma ? std::min(*result.min_sigma, summary.min_sigma) : summary.min_sigma;
    result.max_sigma = result.max_sigma ? std::max(*result.max_sigma, summary.max_sigma) : summary.max_sigma;
  }
}

xt::xtensor<float, 2> build_gaussian_beam_kernel(long halfsize, float sigma_major, float sigma_minor, float rotate)
{
  const auto size = 2 * halfsize + 1;
  xt::xtensor<float, 2> kernel;
  kernel.resize({static_cast<size_t>(size), static_cast<size_t>(size)});

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

namespace
{
/// The weights of the band's channels and the scalars derived from them.
struct BandScalars
{
  xt::xtensor<float, 1> weights_sliced;
  float inv_band_size;
  float weights_sum;
  float inv_weights_sum;
};

BandScalars band_scalars(const PlaneBatchItem &item, const BandSlice &band)
{
  BandScalars scalars;
  scalars.weights_sliced = xt::view(item.weights, band.xrange());
  scalars.inv_band_size = 1.F / static_cast<float>(band.size());
  scalars.weights_sum = std::accumulate(scalars.weights_sliced.cbegin(), scalars.weights_sliced.cend(), 0.F);
  scalars.inv_weights_sum = 1.F / scalars.weights_sum;
  return scalars;
}

/// Weighted image of the band's channels into `scratch.plane`, minus the mean image when the config asks for it.
void band_plane(ProcessBandScratch &scratch, const PlaneBatchItem &item, const PlaneBatchGroup &group,
                const BandSlice &band, const PlaneConfig &config, const BandScalars &scalars)
{
  const size_t n_y = item.stokes_cube.shape(1);
  const size_t n_x = item.stokes_cube.shape(2);
  const auto n_pixels = n_y * n_x;
  const auto &weights = item.weights;
  const bool do_subtract_mean = config.subtract_mean && group.mean_image.has_value();

  auto &plane = scratch.plane;
  plane.resize({n_y, n_x});
  const auto &cube = item.stokes_cube;
  if (cube.shape(1) != n_y || cube.shape(2) != n_x)
  {
    throw std::runtime_error("stokes_cube shape mismatch with plane");
  }
  std::span<const float> mean_image_span;
  if (do_subtract_mean)
  {
    const auto &mean_image = *group.mean_image;
    if (mean_image.shape(1) != n_y || mean_image.shape(2) != n_x)
    {
      throw std::runtime_error("mean_image shape mismatch with plane");
    }
    mean_image_span = std::span<const float>(mean_image.data(), mean_image.size());
  }
  plane_lot::detail::weighted_band_image({cube.data(), cube.size()}, {weights.data(), weights.size()}, mean_image_span,
                                         n_pixels, band.start, band.stop, scalars.inv_weights_sum,
                                         scalars.inv_band_size, {plane.data(), plane.size()});
}

struct NoiseStage
{
  float median_noise;
  float rms_floor; ///< 0 in the global MAD mode
};

/// Noise map of the band into `scratch.noise_map`, from the local RMS or from the global MAD, and its median.
/// `noise_global_mad` returns the global MAD noise of the band's plane (1.4826 times the median of `abs(plane)`), which
/// process_band computes on first use and shares with the detection records.
template <typename GlobalMadFn>
NoiseStage noise_stage(ProcessBandScratch &scratch, const PlaneBatchGroup &group, const BandSlice &band,
                       const PlaneConfig &config, GlobalMadFn &noise_global_mad)
{
  const auto &plane = scratch.plane;
  const size_t n_y = plane.shape(0);
  const size_t n_x = plane.shape(1);

  // Precondition: group.selfnoise_offset/dde_noise must match the plane's shape.
  if (group.selfnoise_offset.shape() != std::array<size_t, 2>{n_y, n_x} ||
      group.dde_noise.shape() != std::array<size_t, 2>{n_y, n_x})
  {
    throw std::runtime_error("selfnoise_offset/dde_noise shape mismatch with plane");
  }
  const std::span<const float> selfnoise_span(group.selfnoise_offset.data(), group.selfnoise_offset.size());
  const std::span<const float> dde_span(group.dde_noise.data(), group.dde_noise.size());

  auto &noise_map = scratch.noise_map;
  noise_map.resize({n_y, n_x});
  const std::span<float> noise_map_span(noise_map.data(), noise_map.size());
  float rms_floor = 0.F;
  if (config.use_rms_noise_estimator)
  {
    // make_noise_map(method='std'): pixels below the map's own median are raised to it.
    const auto &rms = compute_rms_noise_map(scratch, plane, config.boxsize);
    rms_floor = detail::median::radix_median_of({rms.data(), rms.size()});
    plane_lot::detail::combine_noise_map_rms({rms.data(), rms.size()}, rms_floor, selfnoise_span, dde_span,
                                             band.selfnoise_scale_factor, config.ddenoise_scaling_factor,
                                             noise_map_span);
  }
  else
  {
    plane_lot::detail::combine_noise_map_constant(noise_global_mad(), selfnoise_span, dde_span,
                                                  band.selfnoise_scale_factor, config.ddenoise_scaling_factor,
                                                  noise_map_span);
  }
  const float median_noise = detail::median::radix_median_of({noise_map.data(), noise_map.size()});
  return {median_noise, rms_floor};
}

struct BeamKernel
{
  size_t halfsize = 0;
  std::optional<xt::xtensor<float, 2>> kernel; ///< empty when the beam is disabled
};

/// Half size and kernel of the beam for the band: the fixed kernel of the config, or a Gaussian built from the band's
/// weighted beam axes.
BeamKernel beam_kernel_for_band(const PlaneBatchItem &item, const BandSlice &band, const PlaneConfig &config,
                                const BandScalars &scalars)
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

      const auto &weights_sliced = scalars.weights_sliced;
      const auto weights_sum = scalars.weights_sum;
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

/// The per-island quantities that the detection record needs besides the peak, the island size and the noise maps.
struct IslandStatistics
{
  float flux;
  float signal_to_noise_extended_box;
  float beam_chi2;
  double rms; ///< standard deviation of the donut around the island, 0 when it is empty
  float lmad; ///< scaled median absolute value of the donut, 0 when it is empty
  float boost;
};

/// Stamp statistics of island `label` (1 based) of `scratch.islands`, around its peak pixel in `scratch.plane`.
IslandStatistics island_statistics(ProcessBandScratch &scratch, const PlaneConfig &config, const BeamKernel &beam,
                                   size_t label)
{
  const auto &plane = scratch.plane;
  const auto &islands = scratch.islands;
  const size_t n_y = plane.shape(0);
  const size_t n_x = plane.shape(1);
  const size_t radius = config.boxsize / 2;
  const size_t beam_kernel_halfsize = beam.halfsize;

  const auto &island = islands.peak_per_label[label - 1];
  const long peak_y = static_cast<long>(island.coordinates.first);
  const long peak_x = static_cast<long>(island.coordinates.second);

  auto &sub_image = scratch.stamp;
  sub_image.resize({2 * radius + 1, 2 * radius + 1});
  plane_lot::detail::stamp_extract({plane.data(), plane.size()}, n_y, n_x, peak_y, peak_x, static_cast<long>(radius),
                                   {sub_image.data(), sub_image.size()});
  plane_lot::detail::stamp_zero_inner_box({sub_image.data(), sub_image.size()}, static_cast<long>(radius),
                                          config.inner_boxsize);
  const float extreme = plane_lot::detail::stamp_abs_extreme({sub_image.data(), sub_image.size()});
  const auto adjacent_noise = extreme / config.max_to_std_ratio_value();

  const auto flux = plane(island.coordinates.first, island.coordinates.second);
  const auto signal_to_noise_extended_box = adjacent_noise != 0.F ? flux / adjacent_noise : 0.F;

  float beam_chi2_value = 0.F;
  if (beam.kernel.has_value())
  {
    auto &beam_sub_image = scratch.beam_stamp;
    beam_sub_image.resize({2 * beam_kernel_halfsize + 1, 2 * beam_kernel_halfsize + 1});
    plane_lot::detail::stamp_extract({plane.data(), plane.size()}, n_y, n_x, peak_y, peak_x,
                                     static_cast<long>(beam_kernel_halfsize),
                                     {beam_sub_image.data(), beam_sub_image.size()});

    beam_chi2_value = plane_lot::detail::stamp_beam_chi2(
        {beam_sub_image.data(), beam_sub_image.size()}, {beam.kernel->data(), beam.kernel->size()},
        static_cast<long>(beam_kernel_halfsize), adjacent_noise);
  }

  const auto donut =
      plane_lot::detail::stamp_donut_stats({sub_image.data(), sub_image.size()}, static_cast<long>(radius));
  const auto rms = donut.stddev;
  const auto lmad = donut.lmad;

  float boost_value = 0.F;
  if (config.baseline_flux.has_value())
  {
    boost_value = breifast::islands::island_boost({islands.labels.data(), islands.labels.size()},
                                                  {config.baseline_flux->data(), config.baseline_flux->size()}, n_y,
                                                  n_x, label, flux);
  }

  return {flux, signal_to_noise_extended_box, beam_chi2_value, rms, lmad, boost_value};
}
} // namespace

inline void process_band(ProcessBandScratch &scratch, PlaneBatchItem &item, const PlaneBatchGroup &group,
                         const BandSlice &band, const PlaneConfig &config, ProcessPlaneResult &result)
{
  ZoneScoped;
  if (!xt::any(xt::view(item.nonzero_channels, band.xrange())))
  {
    return;
  }

  const size_t n_y = item.stokes_cube.shape(1);
  const size_t n_x = item.stokes_cube.shape(2);
  const auto n_pixels = n_y * n_x;
  if (band.start < 0 || static_cast<size_t>(band.stop) > item.stokes_cube.shape(0))
  {
    throw std::runtime_error("band exceeds the stokes_cube channels");
  }

  // Masking
  auto &image_mask = scratch.image_mask;
  if (group.accept_mask.shape(1) != n_y || group.accept_mask.shape(2) != n_x)
  {
    throw std::runtime_error("accept_mask shape mismatch with plane");
  }
  image_mask.resize({n_y, n_x});
  if (!plane_lot::detail::or_accept_masks(std::span<const bool>(group.accept_mask.data(), group.accept_mask.size()),
                                          n_pixels, band.start, band.stop,
                                          std::span<bool>(image_mask.data(), image_mask.size())))
  {
    return;
  }

  const auto scalars = band_scalars(item, band);
  band_plane(scratch, item, group, band, config, scalars);
  auto &plane = scratch.plane;

  std::optional<float> global_mad_cache;
  const auto noise_global_mad = [&]()
  {
    if (!global_mad_cache)
    {
      global_mad_cache = detail::median::radix_median_of_abs({plane.data(), plane.size()}) * 1.4826F;
    }
    return *global_mad_cache;
  };
  const auto noise = noise_stage(scratch, group, band, config, noise_global_mad);
  auto &noise_map = scratch.noise_map;
  const float median_noise = noise.median_noise;

  result.median_noises.push_back(median_noise);

  compute_snr_and_apply_filters(scratch, plane, noise_map, median_noise, scalars.inv_band_size, image_mask, group, band,
                                config, result);
  if (band.size() == 1 && item.nonzero_channels.size() > 1)
  {
    std::copy_n(scratch.snr.data(), n_pixels, item.stokes_cube.data() + band.start * n_pixels);
  }
  if (!plane_lot::detail::any_true(image_mask.data(), image_mask.size()))
  {
    return;
  }

  const auto beam = beam_kernel_for_band(item, band, config, scalars);

  breifast::islands::label_islands(scratch.islands, image_mask, scratch.snr, noise_map);

  const auto &islands = scratch.islands;

  auto &detection_arrays = result.detections;
  detection_arrays.reserve(result.detections.size() + scratch.islands.peak_per_label.size());
  ZoneNamedN(detections_zone, "detections", true);
  for (size_t label = 1; label <= islands.peak_per_label.size(); ++label)
  {
    const auto &island = islands.peak_per_label[label - 1];
    const auto statistics = island_statistics(scratch, config, beam, label);
    const auto flux = statistics.flux;
    const auto rms = statistics.rms;
    const auto lmad = statistics.lmad;

    const float noisemap0_at_peak =
        config.use_rms_noise_estimator
            ? std::max(scratch.windows_sums(island.coordinates.first, island.coordinates.second), noise.rms_floor)
            : noise_global_mad();

    detection_arrays.add(island.coordinates.first, island.coordinates.second, island.island_size, item.stokes_label,
                         band.label, flux, island.local_noise, island.local_noise / median_noise, island.value,
                         rms != 0.F ? flux / rms : 0.F, statistics.signal_to_noise_extended_box,
                         lmad != 0.F ? flux / lmad : 0.F, flux / noise_global_mad(),
                         group.selfnoise_raw(island.coordinates.first, island.coordinates.second),
                         noise_map(island.coordinates.first, island.coordinates.second) / noisemap0_at_peak,
                         statistics.beam_chi2, statistics.boost);
  }
}

SpectralVarianceStats compute_spectral_variance_stats(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                                      const PlaneConfig &config)
{
  SpectralVarianceStats stats;
  const auto n_y = item.stokes_cube.shape(1);
  const auto n_x = item.stokes_cube.shape(2);

  stats.sum_pow2.resize({n_y, n_x});
  stats.sum_pow3.resize({n_y, n_x});
  stats.sum_pow4.resize({n_y, n_x});

  plane_lot::detail::spectral_variance_moments(
      {item.stokes_cube.data(), item.stokes_cube.size()}, {item.nonzero_channels.data(), item.nonzero_channels.size()},
      n_y * n_x, {stats.sum_pow2.data(), stats.sum_pow2.size()}, {stats.sum_pow3.data(), stats.sum_pow3.size()},
      {stats.sum_pow4.data(), stats.sum_pow4.size()});

  stats.nonzero_channels_mask = item.nonzero_channels;

  return stats;
}

ProcessPlaneResult process_plane(ProcessBandScratch &scratch, const PlaneConfig &config, PlaneBatchItem &item,
                                 const PlaneBatchGroup &group)
{
  ZoneScoped;
  ProcessPlaneResult result;
  for (auto it = config.band_slices.rbegin(); it != config.band_slices.rend(); ++it)
  {
    process_band(scratch, item, group, *it, config, result);
  }
  if (item.nonzero_channels.size() > 1)
  {
    result.variance_stats = compute_spectral_variance_stats(item, group, config);
  }

  return result;
}

std::vector<ProcessPlaneResult> run_batch_core(std::vector<PlaneBatchItem> &items,
                                               const std::vector<PlaneBatchGroup> &groups, const PlaneConfig &config,
                                               int num_threads)
{
  std::vector<ProcessPlaneResult> results(items.size());

  // One task_arena per NUMA node, each pinned to it, so a thread's scratch buffers land on the same
  // socket as the thread (first-touch). Needs libtbbbind, which needs hwloc at TBB build time; without
  // it numa_nodes() returns a single sentinel node and every arena below is silently unconstrained.
  const auto numa_nodes = tbb::info::numa_nodes();

  std::vector<int> node_concurrency(numa_nodes.size());
  int total_natural = 0;
  for (size_t i = 0; i < numa_nodes.size(); ++i)
  {
    node_concurrency[i] = tbb::info::default_concurrency(numa_nodes[i]);
    total_natural += node_concurrency[i];
  }

  // Split num_threads across nodes in proportion to their natural concurrency (largest remainder).
  std::vector<int> arena_concurrency = node_concurrency;
  if (num_threads > 0 && num_threads < total_natural)
  {
    int assigned = 0;
    std::vector<double> remainders(numa_nodes.size());
    for (size_t i = 0; i < numa_nodes.size(); ++i)
    {
      const double share = static_cast<double>(num_threads) * node_concurrency[i] / total_natural;
      arena_concurrency[i] = static_cast<int>(share);
      remainders[i] = share - arena_concurrency[i];
      assigned += arena_concurrency[i];
    }
    std::vector<size_t> order(numa_nodes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return remainders[a] > remainders[b]; });
    for (int i = 0; i < num_threads - assigned; ++i)
    {
      arena_concurrency[order[i]]++;
    }
  }

  std::cerr << "[numa] nodes=" << numa_nodes.size();
  for (size_t i = 0; i < numa_nodes.size(); ++i)
  {
    std::cerr << " [" << numa_nodes[i] << "]=" << arena_concurrency[i];
  }
  std::cerr << std::endl;

  std::vector<tbb::task_arena> arenas;
  arenas.reserve(numa_nodes.size());
  for (size_t i = 0; i < numa_nodes.size(); ++i)
  {
    arenas.emplace_back(
        tbb::task_arena::constraints{}.set_numa_id(numa_nodes[i]).set_max_concurrency(arena_concurrency[i]));
  }

  const int total_concurrency = std::accumulate(arena_concurrency.begin(), arena_concurrency.end(), 0);

  // Fixed contiguous item range per node. Arenas are separate scheduling domains, so this split is
  // also the load-balancing boundary: a node that finishes early cannot steal from a slower one.
  std::vector<size_t> item_offsets(numa_nodes.size() + 1, 0);
  size_t assigned_items = 0;
  for (size_t i = 0; i < numa_nodes.size(); ++i)
  {
    const size_t count = (i + 1 == numa_nodes.size()) ? items.size() - assigned_items
                                                      : items.size() * arena_concurrency[i] / total_concurrency;
    item_offsets[i + 1] = item_offsets[i] + count;
    assigned_items += count;
  }

  // One scratch buffer set per worker thread, reused across every item that thread processes.
  static tbb::enumerable_thread_specific<ProcessBandScratch> scratches;

  const auto main_iteration = [&](const tbb::blocked_range<size_t> &range)
  {
    auto &scratch = scratches.local();
    for (auto index = range.begin(); index != range.end(); ++index)
    {
      auto &item = items[index];
      const auto group_index = item.group_index;
      const auto &group = groups[group_index];

      try
      {
        results[index] = process_plane(scratch, config, item, group);
      }
      catch (const std::exception &e)
      {
        std::cerr << e.what() << std::endl;
      }
    }
  };
  const auto run_slice = [&](size_t node)
  {
    arenas[node].execute(
        [&]()
        { tbb::parallel_for(tbb::blocked_range<size_t>(item_offsets[node], item_offsets[node + 1]), main_iteration); });
  };

  __itt_resume();
  tbb::task_group task_group;
  for (size_t node = 0; node + 1 < arenas.size(); ++node)
  {
    task_group.run([&, node]() { run_slice(node); });
  }
  run_slice(arenas.size() - 1); // the calling thread takes the last node's slice instead of sitting idle
  task_group.wait();
  __itt_pause();

  return results;
}

} // namespace breifast