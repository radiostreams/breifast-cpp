#include "islands.hpp"
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

inline xt::xarray<float> compute_rms_noise_map(const xt::xarray<float> &plane, long boxsize)
{
  const long n_y = static_cast<long>(plane.shape(0));
  const long n_x = static_cast<long>(plane.shape(1));

  xt::xarray<float> parts_sums;
  parts_sums.resize({static_cast<size_t>(n_y + 1), static_cast<size_t>(n_x + 1)});
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
  windows_sums.resize({static_cast<size_t>(n_y), static_cast<size_t>(n_x)});

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

xt::xarray<float> compute_snr_and_apply_filters(const xt::xarray<float> &plane, const xt::xarray<float> &noise_map,
                                                float median_noise, float inv_band_size, xt::xarray<bool> &image_mask,
                                                const PlaneBatchGroup &group, const BandSlice &band,
                                                const PlaneConfig &config, ProcessPlaneResult &result)
{
  const size_t n_y = plane.shape(0);
  const size_t n_x = plane.shape(1);

  xt::xarray<float> snr;
  snr.resize(plane.shape());
  result.histogram0.clear();
  result.histogram0.resize(config.max_sigma_bin * 2, 0);
  result.histogram1.clear();
  result.histogram1.resize(config.max_sigma_bin * 2, 0);
  result.histogram2.clear();
  result.histogram2.resize(config.max_sigma_bin * 2, 0);
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

xt::xarray<float> build_gaussian_beam_kernel(long halfsize, float sigma_major, float sigma_minor, float rotate)
{
  const auto size = 2 * halfsize + 1;
  xt::xarray<float> kernel;
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
  for (const auto &[label, island] : islands.peak_per_label)
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

SpectralVarianceStats compute_spectral_variance_stats(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                                      const PlaneConfig &config)
{
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

ProcessPlaneResult process_plane(const PlaneConfig &config, PlaneBatchItem &item, const PlaneBatchGroup &group)
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

std::vector<ProcessPlaneResult> run_batch_core(std::vector<PlaneBatchItem> &items,
                                               const std::vector<PlaneBatchGroup> &groups, const PlaneConfig &config,
                                               int num_threads)
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

} // namespace breifast