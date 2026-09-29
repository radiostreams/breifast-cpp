#include <algorithm>
#include <ittnotify.h>
#include <span>
#include <tracy/Tracy.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/reducers/xreducer.hpp>

#include "islands.hpp"
#include "median.hpp"
#include "plane_lot_dispatch.hpp"

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

inline const xt::xtensor<float, 2> &compute_rms_noise_map(ProcessBandScratch &scratch,
                                                          const xt::xtensor<float, 2> &plane, long boxsize)
{
  ZoneScoped;
  const long n_y = static_cast<long>(plane.shape(0));
  const long n_x = static_cast<long>(plane.shape(1));

  const long radius = boxsize / 2;
  const long window_size = std::max(boxsize, 1L);

  const auto min_x_of = [&](long i_x) { return std::max(i_x - radius, 0L); };
  const auto max_x_of = [&](long i_x) { return std::min(i_x + window_size - radius, n_x); };
  const auto min_y_of = [&](long i_y) { return std::max(i_y - radius, 0L); };
  const auto max_y_of = [&](long i_y) { return std::min(i_y + window_size - radius, n_y); };

  auto &windows_sums = scratch.windows_sums;
  windows_sums.resize({static_cast<size_t>(n_y), static_cast<size_t>(n_x)});

  auto &row_window_sums = scratch.row_window_sums;
  row_window_sums.resize({static_cast<size_t>(n_y), static_cast<size_t>(n_x)});

  for (long i_y = 0; i_y < n_y; ++i_y)
  {
    double accumulator = 0.;
    for (long i_x = 0; i_x < max_x_of(0); ++i_x)
    {
      const double value = plane(i_y, i_x);
      accumulator += value * value;
    }

    for (long i_x = 0; i_x < n_x; ++i_x)
    {
      row_window_sums(i_y, i_x) = accumulator;

      const auto entering_column = i_x + window_size - radius;
      const auto leaving_column = i_x - radius;
      if (entering_column < n_x)
      {
        const double value = plane(i_y, entering_column);
        accumulator += value * value;
      }

      if (leaving_column >= 0)
      {
        const double value = plane(i_y, leaving_column);
        accumulator -= value * value;
      }
    }
  }

  static const long block_size = []
  {
    const char *env = std::getenv("BREIFAST_BLOCK_SIZE");
    return env != nullptr ? std::strtol(env, nullptr, 10) : 512L;
  }();
  auto &column_accumulators = scratch.column_accumulators;
  column_accumulators.resize(static_cast<size_t>(block_size));

  for (long block_start = 0; block_start < n_x; block_start += block_size)
  {
    const long block_width = std::min(block_start + block_size, n_x) - block_start;

    std::fill_n(column_accumulators.begin(), block_width, 0.);

    for (long i_y = 0; i_y < max_y_of(0); ++i_y)
    {
      for (long i = 0; i < block_width; ++i)
      {
        column_accumulators[i] += row_window_sums(i_y, block_start + i);
      }
    }

    for (long i_y = 0; i_y < n_y; ++i_y)
    {
      const long rows_in_window = max_y_of(i_y) - min_y_of(i_y);
      for (long i = 0; i < block_width; ++i)
      {
        const long i_x = block_start + i;
        const long count = rows_in_window * (max_x_of(i_x) - min_x_of(i_x));
        windows_sums(i_y, i_x) = static_cast<float>(std::sqrt(column_accumulators[i] / static_cast<double>(count)));
      }

      const long entering_row = i_y + window_size - radius;
      const long leaving_row = i_y - radius;
      if (entering_row < n_y)
      {
        for (long i = 0; i < block_width; ++i)
        {
          column_accumulators[i] += row_window_sums(entering_row, block_start + i);
        }
      }
      if (leaving_row >= 0)
      {
        for (long i = 0; i < block_width; ++i)
        {
          column_accumulators[i] -= row_window_sums(leaving_row, block_start + i);
        }
      }
    }
  }

  const auto median = detail::median::radix_median_of({windows_sums.data(), windows_sums.size()});
  for (long i_y = 0; i_y < n_y; ++i_y)
  {
    for (long i_x = 0; i_x < n_x; ++i_x)
    {
      windows_sums(i_y, i_x) = std::max(windows_sums(i_y, i_x), median);
    }
  }

  return windows_sums;
}

void compute_snr_and_apply_filters(ProcessBandScratch &scratch, const xt::xtensor<float, 2> &plane,
                                   const xt::xtensor<float, 2> &noise_map, float median_noise, float inv_band_size,
                                   xt::xtensor<bool, 2> &image_mask, const PlaneBatchGroup &group,
                                   const BandSlice &band, const PlaneConfig &config, ProcessPlaneResult &result)
{
  ZoneScoped;
  const size_t n_y = plane.shape(0);
  const size_t n_x = plane.shape(1);

  scratch.snr.resize(plane.shape());
  scratch.histograms[0].resize(config.max_sigma_bin * 2, 0);
  scratch.histograms[1].resize(config.max_sigma_bin * 2, 0);
  scratch.histograms[2].resize(config.max_sigma_bin * 2, 0);
  std::fill(scratch.histograms[0].begin(), scratch.histograms[0].end(), 0);
  std::fill(scratch.histograms[1].begin(), scratch.histograms[1].end(), 0);
  std::fill(scratch.histograms[2].begin(), scratch.histograms[2].end(), 0);

  float local_min_sigma = std::numeric_limits<float>::infinity();
  float local_max_sigma = std::numeric_limits<float>::lowest();

  const float *const plane_ptr = plane.data();
  const float *const noise_map_ptr = noise_map.data();
  float *const snr_ptr = scratch.snr.data();
  bool *const image_mask_ptr = image_mask.data();
  const bool snr_cutoff_enabled = group.snr_cutoff.has_value();
  const float *const snr_cutoff_ptr = snr_cutoff_enabled ? group.snr_cutoff->data() : nullptr;
  const bool flux_cutoff_enabled = group.flux_cutoff.has_value();
  const float *const flux_cutoff_ptr = flux_cutoff_enabled ? group.flux_cutoff->data() : nullptr;

  for (size_t i_pixel = 0; i_pixel < n_y * n_x; ++i_pixel)
  {
    const auto noise = noise_map_ptr[i_pixel];
    const bool valid = noise != 0 && std::isfinite(noise);
    snr_ptr[i_pixel] = valid ? plane_ptr[i_pixel] / noise : 0;
    image_mask_ptr[i_pixel] &= valid;

    const auto mask0 = image_mask_ptr[i_pixel];

    float snr_cutoff_threshold = 0.F;
    if (snr_cutoff_enabled)
    {
      for (auto i_channel = band.start; i_channel < band.stop; ++i_channel)
      {
        snr_cutoff_threshold += snr_cutoff_ptr[i_channel * n_y * n_x + i_pixel];
      }
      snr_cutoff_threshold *= inv_band_size;
    }

    float flux_cutoff_threshold = 0.F;
    if (flux_cutoff_enabled)
    {
      for (auto i_channel = band.start; i_channel < band.stop; ++i_channel)
      {
        flux_cutoff_threshold += flux_cutoff_ptr[i_channel * n_y * n_x + i_pixel];
      }
      flux_cutoff_threshold *= inv_band_size;
    }

    const auto mask1 = mask0 && (!snr_cutoff_enabled || std::abs(snr_ptr[i_pixel]) >= snr_cutoff_threshold);
    const auto mask2 = mask1 && (!flux_cutoff_enabled || std::abs(plane_ptr[i_pixel]) >= flux_cutoff_threshold);
    const auto mask3 = mask2 && (config.reject_noise_threshold == 0.F ||
                                 noise_map_ptr[i_pixel] <= median_noise * config.reject_noise_threshold);

    image_mask_ptr[i_pixel] = mask3;

    long bin_index = std::floor(snr_ptr[i_pixel] + config.max_sigma_bin);
    if (bin_index == 2 * config.max_sigma_bin)
    {
      bin_index = 2 * config.max_sigma_bin - 1;
    }
    if (0 <= bin_index && bin_index < 2 * config.max_sigma_bin)
    {
      if (mask0 && config.histogram_detail > 1)
      {
        scratch.histograms[0][bin_index]++;
      }
      if (mask1 && config.histogram_detail > 1)
      {
        scratch.histograms[1][bin_index]++;
      }
      if (mask2 && config.histogram_detail > 0)
      {
        scratch.histograms[2][bin_index]++;
      }
    }

    if (mask3)
    {
      const float value = snr_ptr[i_pixel];
      local_min_sigma = std::min(local_min_sigma, value);
      local_max_sigma = std::max(local_max_sigma, value);
    }

    const float thresholded_abs_snr = mask3 ? std::abs(snr_ptr[i_pixel]) : 0.F;
    image_mask_ptr[i_pixel] = thresholded_abs_snr >= config.threshold;
  }
  result.histograms = scratch.histograms;
  if (local_min_sigma < std::numeric_limits<float>::infinity())
  {
    result.min_sigma = result.min_sigma ? std::min(*result.min_sigma, local_min_sigma) : local_min_sigma;
    result.max_sigma = result.max_sigma ? std::max(*result.max_sigma, local_max_sigma) : local_max_sigma;
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

  // Masking
  auto &image_mask = scratch.image_mask;
  bool any_mask = false;
  {
    ZoneScopedN("mask_accumulate");
    xt::noalias(image_mask) = xt::view(group.accept_mask, band.start, xt::all(), xt::all());
    any_mask = xt::any(image_mask);
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
  }

  if (!any_mask)
  {
    return;
  }

  const auto &weights = item.weights;
  const xt::xtensor<float, 1> weights_sliced = xt::view(weights, band.xrange());

  const bool do_subtract_mean = config.subtract_mean && group.mean_image.has_value();
  const float inv_band_size = 1.F / static_cast<float>(band.size());

  const float weights_sum = std::accumulate(weights_sliced.cbegin(), weights_sliced.cend(), 0.F);
  const float inv_weights_sum = 1.F / weights_sum;

  auto &plane = scratch.plane;
  {
    ZoneScopedN("band_accumulate");
    plane.resize({n_y, n_x});
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
  }

  const float noise_global_mad = detail::median::radix_median_of_abs({plane.data(), plane.size()}) * 1.4826F;

  const auto combine = [&](auto &noisemap, const auto &noisemap0)
  {
    ZoneScopedN("combine");
    // Precondition: group.selfnoise_offset/dde_noise must match the plane's shape.
    if (group.selfnoise_offset.shape() != std::array<size_t, 2>{n_y, n_x} ||
        group.dde_noise.shape() != std::array<size_t, 2>{n_y, n_x})
    {
      throw std::runtime_error("selfnoise_offset/dde_noise shape mismatch with plane");
    }

    constexpr bool noisemap0_is_scalar = std::is_floating_point_v<std::decay_t<decltype(noisemap0)>>;
    const auto noisemap0_at = [&](size_t i_y, size_t i_x) -> float
    {
      if constexpr (noisemap0_is_scalar)
      {
        return noisemap0;
      }
      else
      {
        return noisemap0(i_y, i_x);
      }
    };

    noisemap.resize({n_y, n_x});

    for (size_t i_y = 0; i_y < n_y; ++i_y)
    {
      for (size_t i_x = 0; i_x < n_x; ++i_x)
      {
        const float noisemap0_value = noisemap0_at(i_y, i_x);
        noisemap(i_y, i_x) = std::sqrt(noisemap0_value * noisemap0_value +
                                       band.selfnoise_scale_factor * group.selfnoise_offset(i_y, i_x) +
                                       config.ddenoise_scaling_factor * group.dde_noise(i_y, i_x));
      }
    }
    const float mednoise = detail::median::radix_median_of({noisemap.data(), noisemap.size()});
    return mednoise;
  };

  auto &noise_map = scratch.noise_map;
  float median_noise = 0.F;
  if (config.use_rms_noise_estimator)
  {
    median_noise = combine(scratch.noise_map, compute_rms_noise_map(scratch, plane, config.boxsize));
  }
  else
  {
    median_noise = combine(scratch.noise_map, noise_global_mad);
  }
  result.median_noises.push_back(median_noise);

  compute_snr_and_apply_filters(scratch, plane, noise_map, median_noise, inv_band_size, image_mask, group, band, config,
                                result);
  if (band.size() == 1)
  {
    xt::view(item.stokes_cube, band.start) = scratch.snr;
  }
  if (!xt::any(image_mask))
  {
    return;
  }

  const size_t radius = config.boxsize / 2;
  size_t pad;
  size_t beam_kernel_halfsize = 0;
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

  auto &padded_image = scratch.padded_image;
  {
    ZoneScopedN("pad_image");
    padded_image.resize({plane.shape(0) + pad * 2, plane.shape(1) + pad * 2});
    const size_t padded_ny = static_cast<size_t>(padded_image.shape(0));
    const size_t padded_nx = static_cast<size_t>(padded_image.shape(1));
    for (size_t i_y = 0; i_y < padded_ny; ++i_y)
    {
      for (size_t i_x = 0; i_x < padded_nx; ++i_x)
      {
        const bool in_padding = i_y < pad || i_x < pad || i_y >= pad + n_y || i_x >= pad + n_x;
        padded_image(i_y, i_x) = in_padding ? 0 : plane(i_y - pad, i_x - pad);
      }
    }
  }

  breifast::islands::label_islands(scratch.islands, image_mask, scratch.snr, noise_map);

  const auto &islands = scratch.islands;
  std::optional<xt::xtensor<float, 2>> beam_kernel;
  if (config.beam_size_enabled)
  {
    if (config.fixed_beam_kernel.has_value())
    {
      beam_kernel = xt::xtensor<float, 2>(*config.fixed_beam_kernel);
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

  auto &detection_arrays = result.detections;
  detection_arrays.reserve(result.detections.size() + scratch.islands.peak_per_label.size());
  ZoneNamedN(detections_zone, "detections", true);
  for (size_t label = 1; label <= islands.peak_per_label.size(); ++label)
  {
    const auto &island = islands.peak_per_label[label - 1];
    const auto padded_y = island.coordinates.first + pad;
    const auto padded_x = island.coordinates.second + pad;

    xt::xtensor<float, 2> sub_image = xt::view(padded_image, xt::range(padded_y - radius, padded_y + radius + 1),
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
      xt::xtensor<float, 2> beam_sub_image =
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
    const auto rms = donut.size() == 0 ? 0.F : xt::stddev(donut)();
    const auto lmad = donut.size() == 0 ? 0.F : xt::median(xt::abs(donut)) * 1.4826F;

    float boost_value = 0.F;
    if (config.baseline_flux.has_value())
    {
      const auto island_mask = xt::equal(islands.labels, label);
      const xt::xtensor<float, 1> ratio = flux / xt::filter(*config.baseline_flux, island_mask);
      const auto max_abs_index = xt::argmax(xt::abs(ratio))();
      boost_value = ratio(max_abs_index);
    }

    const float noisemap0_at_peak = config.use_rms_noise_estimator
                                        ? scratch.windows_sums(island.coordinates.first, island.coordinates.second)
                                        : noise_global_mad;

    detection_arrays.add(island.coordinates.first, island.coordinates.second, island.island_size, item.stokes_label,
                         band.label, flux, island.local_noise, island.local_noise / median_noise, island.value,
                         rms != 0.F ? flux / rms : 0.F, signal_to_noise_extended_box, lmad != 0.F ? flux / lmad : 0.F,
                         flux / noise_global_mad,
                         group.selfnoise_raw(island.coordinates.first, island.coordinates.second),
                         noise_map(island.coordinates.first, island.coordinates.second) / noisemap0_at_peak,
                         beam_chi2_value, boost_value);
  }
}

SpectralVarianceStats compute_spectral_variance_stats(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                                      const PlaneConfig &config)
{
  ZoneScoped;
  SpectralVarianceStats stats;
  const auto n_channels = item.stokes_cube.shape(0);
  const auto n_y = item.stokes_cube.shape(1);
  const auto n_x = item.stokes_cube.shape(2);

  stats.sum_pow2.resize({n_y, n_x});
  stats.sum_pow3.resize({n_y, n_x});
  stats.sum_pow4.resize({n_y, n_x});

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