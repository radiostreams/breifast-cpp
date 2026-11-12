#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>

#include "gpu_band.hpp"
#include "gpu_plane.hpp"
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
  const auto &islands = band.islands;
  assert(std::apply([count](const auto &...column) { return ((column.size() >= count) && ...); }, islands.tie()));

  auto &detections = result.detections;
  detections.reserve(detections.size() + count);
  for (size_t j = 0; j < count; ++j)
  {
    const float flux = islands.flux[j];
    const float rms = islands.donut_std[j];
    const float lmad = islands.donut_lmad[j];
    const float adjacent_noise = islands.stamp_extreme[j] / max_to_std_ratio;
    const float noisemap0_at_peak =
        use_rms_noise_estimator ? std::max(islands.rms_at_peak[j], band.rms_floor) : band.global_mad;

    detections.add(islands.peak_y[j], islands.peak_x[j], static_cast<long>(islands.island_size[j]), labels.stokes_label,
                   labels.band_label, flux, islands.peak_local_noise[j],
                   islands.peak_local_noise[j] / band.median_noise, islands.peak_value[j],
                   rms != 0.F ? flux / rms : 0.F, adjacent_noise != 0.F ? flux / adjacent_noise : 0.F,
                   lmad != 0.F ? flux / lmad : 0.F, flux / band.global_mad, islands.selfnoise_at_peak[j],
                   islands.noise_at_peak[j] / noisemap0_at_peak, islands.beam_chi2[j], islands.boost[j]);
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
                                                                    cudaStream_t stream)
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

/// Copies `array` into `buffer`, growing it if needed, and returns the device pointer.
template <typename T, typename Array> const T *upload(DeviceVector<T> &buffer, const Array &array)
{
  buffer.assign(array.data(), array.size());
  return buffer.data();
}

/// Same for an optional input: null when it is absent.
template <typename T, typename Array> const T *upload(DeviceVector<T> &buffer, const std::optional<Array> &array)
{
  return array ? upload(buffer, *array) : nullptr;
}
} // namespace

ProcessPlaneResult gpu_process_plane(GpuSlot &slot, const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                     const PlaneConfig &config)
{
  const std::size_t n_channels = item.stokes_cube.shape(0);
  const std::size_t n_y = item.stokes_cube.shape(1);
  const std::size_t n_x = item.stokes_cube.shape(2);

  // The uploads run on the default stream, which the slot's stream does not wait for: it must be idle.
  slot.stream.synchronize();
  auto &inputs = slot.inputs;
  const GpuPlaneInput plane{
      .cube = upload(inputs.cube, item.stokes_cube),
      .weights = upload(inputs.weights, item.weights),
      .accept_mask = upload(inputs.accept_mask, group.accept_mask),
      .mean_image = config.subtract_mean ? upload(inputs.mean_image, group.mean_image) : nullptr,
      .selfnoise_offset = upload(inputs.selfnoise_offset, group.selfnoise_offset),
      .selfnoise_raw = upload(inputs.selfnoise_raw, group.selfnoise_raw),
      .dde_noise = upload(inputs.dde_noise, group.dde_noise),
      .snr_cutoff = upload(inputs.snr_cutoff, group.snr_cutoff),
      .flux_cutoff = upload(inputs.flux_cutoff, group.flux_cutoff),
      .baseline_flux = upload(inputs.baseline_flux, config.baseline_flux),
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
    const bool *const nonzero_channels = upload(inputs.nonzero_channels, item.nonzero_channels);
    result.variance_stats =
        gpu_compute_spectral_variance_stats(plane.cube, nonzero_channels, n_y, n_x, n_channels, slot.stream);
  }

  // Same order as the CPU process_plane: the last band first.
  const bool *const nonzero = item.nonzero_channels.data();
  for (auto slice = config.band_slices.rbegin(); slice != config.band_slices.rend(); ++slice)
  {
    if (std::none_of(nonzero + slice->start, nonzero + slice->stop, std::identity{}))
    {
      continue;
    }
    const auto beam = beam_kernel_for_band(item, *slice, config);
    const auto beam_halfsize = static_cast<long>(beam.halfsize);
    const GpuBand band{
        .start = slice->start,
        .stop = slice->stop,
        .selfnoise_scale_factor = slice->selfnoise_scale_factor,
        .beam_kernel = upload(inputs.beam_kernel, beam.kernel),
        .beam_halfsize = beam_halfsize,
        .beam_kernel_sum = beam.kernel ? beam_kernel_sum(beam.kernel->data(), beam_halfsize) : 0.F,
    };
    if (!gpu_process_band(plane, band, slot.workspace, slot.result, slot.stream))
    {
      continue;
    }
    copy_band_result(slot.result, slot.host, slot.stream);
    append_band_result(slot.host, {item.stokes_label, slice->label}, plane.max_to_std_ratio,
                       config.use_rms_noise_estimator, result);
  }
  return result;
}

ProcessPlaneResult gpu_process_plane(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                     const PlaneConfig &config)
{
  GpuSlot slot;
  return gpu_process_plane(slot, item, group, config);
}
} // namespace breifast::gpu
