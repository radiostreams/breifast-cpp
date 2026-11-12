#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>

#include "gpu_plane_detections.hpp"
#include "gpu_plane_integration.hpp"
#include "gpu_spectral_variance.hpp"

namespace breifast::gpu
{
void append_band_result(const GpuBandHostResult &band, const GpuBandLabels &labels, float max_to_std_ratio,
                        bool use_rms_noise_estimator, breifast::ProcessPlaneResult &result)
{
  assert(max_to_std_ratio >= 0.F);

  const std::array<const std::vector<unsigned long long> *, 3> histograms{&band.histogram0, &band.histogram1,
                                                                          &band.histogram2};
  for (size_t i = 0; i < result.histograms.size(); ++i)
  {
    result.histograms[i].resize(histograms[i]->size());
    std::transform(histograms[i]->begin(), histograms[i]->end(), result.histograms[i].begin(),
                   [](const unsigned long long value) { return static_cast<long>(value); });
  }
  result.median_noises.push_back(band.median_noise);
  if (band.min_sigma < std::numeric_limits<float>::infinity())
  {
    result.min_sigma = result.min_sigma ? std::min(*result.min_sigma, band.min_sigma) : band.min_sigma;
    result.max_sigma = result.max_sigma ? std::max(*result.max_sigma, band.max_sigma) : band.max_sigma;
  }

  const size_t count = band.label_count;
  [[maybe_unused]] const auto has_count = [count](const auto &values) { return values.size() >= count; };
  assert(has_count(band.peak_value) && has_count(band.peak_local_noise) && has_count(band.peak_y) &&
         has_count(band.peak_x) && has_count(band.island_size) && has_count(band.stamp_extreme) &&
         has_count(band.donut_std) && has_count(band.donut_lmad) && has_count(band.beam_chi2) && has_count(band.flux) &&
         has_count(band.boost) && has_count(band.noise_at_peak) && has_count(band.rms_at_peak) &&
         has_count(band.selfnoise_at_peak));

  auto &detections = result.detections;
  detections.reserve(detections.size() + count);
  for (size_t j = 0; j < count; ++j)
  {
    const float flux = band.flux[j];
    const float rms = band.donut_std[j];
    const float lmad = band.donut_lmad[j];
    const float adjacent_noise = band.stamp_extreme[j] / max_to_std_ratio;
    const float noisemap0_at_peak =
        use_rms_noise_estimator ? std::max(band.rms_at_peak[j], band.rms_floor) : band.global_mad;

    detections.add(band.peak_y[j], band.peak_x[j], static_cast<long>(band.island_size[j]), labels.stokes_label,
                   labels.band_label, flux, band.peak_local_noise[j], band.peak_local_noise[j] / band.median_noise,
                   band.peak_value[j], rms != 0.F ? flux / rms : 0.F,
                   adjacent_noise != 0.F ? flux / adjacent_noise : 0.F, lmad != 0.F ? flux / lmad : 0.F,
                   flux / band.global_mad, band.selfnoise_at_peak[j], band.noise_at_peak[j] / noisemap0_at_peak,
                   band.beam_chi2[j], band.boost[j]);
  }
}

breifast::SpectralVarianceStats assemble_spectral_variance_stats(const GpuSpectralVarianceHostResult &host, size_t n_y,
                                                                 size_t n_x, size_t n_channels)
{
  assert(n_y > 0 && n_x > 0 && n_channels > 0);
  [[maybe_unused]] const size_t n_pixels = n_y * n_x;
  assert(host.sum_pow2.size() == n_pixels && host.sum_pow3.size() == n_pixels && host.sum_pow4.size() == n_pixels);
  assert(host.nonzero_channels_mask.size() == n_channels);

  breifast::SpectralVarianceStats stats;
  stats.sum_pow2.resize({n_y, n_x});
  stats.sum_pow3.resize({n_y, n_x});
  stats.sum_pow4.resize({n_y, n_x});
  stats.nonzero_channels_mask.resize({n_channels});

  std::copy(host.sum_pow2.begin(), host.sum_pow2.end(), stats.sum_pow2.begin());
  std::copy(host.sum_pow3.begin(), host.sum_pow3.end(), stats.sum_pow3.begin());
  std::copy(host.sum_pow4.begin(), host.sum_pow4.end(), stats.sum_pow4.begin());
  std::transform(host.nonzero_channels_mask.begin(), host.nonzero_channels_mask.end(),
                 stats.nonzero_channels_mask.begin(), [](const unsigned char value) { return value != 0; });
  return stats;
}

breifast::SpectralVarianceStats gpu_compute_spectral_variance_stats(const float *cube, const bool *nonzero_channels,
                                                                    size_t n_y, size_t n_x, size_t n_channels,
                                                                    GpuStream stream)
{
  GpuSpectralVarianceHostResult host;
  gpu_spectral_variance_stats_host(cube, nonzero_channels, n_y, n_x, n_channels, host, stream);
  return assemble_spectral_variance_stats(host, n_y, n_x, n_channels);
}

namespace
{
/// Sum of a (2 * halfsize + 1)^2 beam kernel, with the same xtensor expression as the CPU stamp_beam_chi2, so that both
/// backends divide by the same float.
float beam_kernel_sum(const float *kernel, long halfsize)
{
  const auto side = static_cast<std::size_t>(2 * halfsize + 1);
  return xt::sum(xt::adapt(kernel, side * side, xt::no_ownership(), std::array<std::size_t, 2>{side, side}))();
}

/// Device copies of one plane's inputs; an absent optional input stays empty.
struct GpuPlaneBuffers
{
  DeviceVector<float> cube;
  DeviceVector<float> weights;
  DeviceVector<bool> nonzero_channels;
  DeviceVector<bool> accept_mask;
  DeviceVector<float> selfnoise_offset;
  DeviceVector<float> selfnoise_raw;
  DeviceVector<float> dde_noise;
  std::optional<DeviceVector<float>> mean_image;
  std::optional<DeviceVector<float>> snr_cutoff;
  std::optional<DeviceVector<float>> flux_cutoff;
  std::optional<DeviceVector<float>> baseline_flux;
};

template <typename Array> auto to_device(const Array &array)
{
  return DeviceVector<typename Array::value_type>::from_host(array.data(), array.size());
}

template <typename Array> auto to_device(const std::optional<Array> &array)
{
  std::optional<DeviceVector<typename Array::value_type>> vector;
  if (array)
  {
    vector = to_device(*array);
  }
  return vector;
}

template <typename T> const T *data_or_null(const std::optional<DeviceVector<T>> &vector)
{
  return vector ? vector->data() : nullptr;
}
} // namespace

ProcessPlaneResult gpu_process_plane_batch(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                           const PlaneConfig &config)
{
  const std::size_t n_channels = item.stokes_cube.shape(0);
  const std::size_t n_y = item.stokes_cube.shape(1);
  const std::size_t n_x = item.stokes_cube.shape(2);

  const GpuPlaneBuffers buffers{
      .cube = to_device(item.stokes_cube),
      .weights = to_device(item.weights),
      .nonzero_channels = to_device(item.nonzero_channels),
      .accept_mask = to_device(group.accept_mask),
      .selfnoise_offset = to_device(group.selfnoise_offset),
      .selfnoise_raw = to_device(group.selfnoise_raw),
      .dde_noise = to_device(group.dde_noise),
      .mean_image = config.subtract_mean ? to_device(group.mean_image) : std::nullopt,
      .snr_cutoff = to_device(group.snr_cutoff),
      .flux_cutoff = to_device(group.flux_cutoff),
      .baseline_flux = to_device(config.baseline_flux),
  };
  const GpuPlaneInput plane{
      .cube = buffers.cube.data(),
      .weights = buffers.weights.data(),
      .accept_mask = buffers.accept_mask.data(),
      .mean_image = data_or_null(buffers.mean_image),
      .selfnoise_offset = buffers.selfnoise_offset.data(),
      .selfnoise_raw = buffers.selfnoise_raw.data(),
      .dde_noise = buffers.dde_noise.data(),
      .snr_cutoff = data_or_null(buffers.snr_cutoff),
      .flux_cutoff = data_or_null(buffers.flux_cutoff),
      .baseline_flux = data_or_null(buffers.baseline_flux),
      .n_y = n_y,
      .n_x = n_x,
      .n_channels = n_channels,
      .boxsize = config.boxsize,
      .inner_boxsize = config.inner_boxsize,
      .histogram_detail = config.histogram_detail,
      .max_sigma_bin = config.max_sigma_bin,
      .ddenoise_scaling_factor = config.ddenoise_scaling_factor,
      .reject_noise_threshold = config.reject_noise_threshold,
      .threshold = config.threshold,
      .max_to_std_ratio = config.max_to_std_ratio_value(),
      .use_rms_noise_estimator = config.use_rms_noise_estimator,
  };

  ProcessPlaneResult result;
  if (n_channels > 1)
  {
    result.variance_stats =
        gpu_compute_spectral_variance_stats(buffers.cube, buffers.nonzero_channels, n_y, n_x, n_channels);
  }

  // Same order as the CPU process_plane: the last band first.
  const Stream stream;
  GpuBandHostResult host;
  const bool *const nonzero = item.nonzero_channels.data();
  for (auto slice = config.band_slices.rbegin(); slice != config.band_slices.rend(); ++slice)
  {
    if (std::none_of(nonzero + slice->start, nonzero + slice->stop, std::identity{}))
    {
      continue;
    }
    const auto beam = beam_kernel_for_band(item, *slice, config);
    std::optional<DeviceVector<float>> beam_kernel;
    if (beam.kernel)
    {
      beam_kernel = to_device(*beam.kernel);
    }
    const auto beam_halfsize = static_cast<long>(beam.halfsize);
    const GpuBand band{
        .start = slice->start,
        .stop = slice->stop,
        .selfnoise_scale_factor = slice->selfnoise_scale_factor,
        .beam_kernel = data_or_null(beam_kernel),
        .beam_halfsize = beam_halfsize,
        .beam_kernel_sum = beam.kernel ? beam_kernel_sum(beam.kernel->data(), beam_halfsize) : 0.F,
    };
    GpuBandWorkspace workspace(band_workspace_sizes(plane, band)); // allocated per band, as before, until 13c
    GpuBandResult band_result(plane.n_pixels(), band_workspace_sizes(plane, band).histogram_bins);
    if (!gpu_process_band(plane, band, workspace, band_result, stream.get()))
    {
      continue;
    }
    copy_band_result(band_result, host, stream.get());
    append_band_result(host, {item.stokes_label, slice->label}, plane.max_to_std_ratio, config.use_rms_noise_estimator,
                       result);
  }
  return result;
}
} // namespace breifast::gpu
