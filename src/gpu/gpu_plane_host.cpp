#include <algorithm>
#include <array>
#include <cassert>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>

#include "cpu/beam_kernel.hpp"
#include "gpu_plane_integration.hpp"
#include "gpu_plane_jobs.hpp"
#include "gpu_plane_result.hpp"
#include "gpu_spectral_variance.hpp"

namespace breifast::gpu
{
void assemble_plane_detections(const GpuPlaneHostResult &plane, const std::vector<GpuBandLabels> &labels,
                               float max_to_std_ratio, bool use_rms_noise_estimator,
                               breifast::DetectionArrays &detections)
{
  assert(labels.size() == plane.statuses.size() && labels.size() == plane.bands.size());
  assert(max_to_std_ratio >= 0.F);

  for (size_t i = plane.statuses.size(); i > 0; --i)
  {
    const size_t band_index = i - 1;
    if (plane.statuses[band_index] == GpuBandStatus::skipped)
    {
      continue;
    }

    const auto &band = plane.bands[band_index];
    const size_t count = band.label_count;
    [[maybe_unused]] const auto has_count = [count](const auto &values) { return values.size() >= count; };
    assert(has_count(band.peak_value) && has_count(band.peak_local_noise) && has_count(band.peak_y) &&
           has_count(band.peak_x) && has_count(band.island_size) && has_count(band.stamp_extreme) &&
           has_count(band.donut_std) && has_count(band.donut_lmad) && has_count(band.beam_chi2) &&
           has_count(band.flux) && has_count(band.boost) && has_count(band.noise_at_peak) &&
           has_count(band.rms_at_peak) && has_count(band.selfnoise_at_peak));

    detections.reserve(detections.size() + count);
    for (size_t j = 0; j < count; ++j)
    {
      const float flux = band.flux[j];
      const float rms = band.donut_std[j];
      const float lmad = band.donut_lmad[j];
      const float adjacent_noise = band.stamp_extreme[j] / max_to_std_ratio;
      const float noisemap0_at_peak =
          use_rms_noise_estimator ? std::max(band.rms_at_peak[j], band.rms_floor) : band.global_mad;

      detections.add(band.peak_y[j], band.peak_x[j], static_cast<long>(band.island_size[j]),
                     labels[band_index].stokes_label, labels[band_index].band_label, flux,
                     band.peak_local_noise[j], band.peak_local_noise[j] / band.median_noise, band.peak_value[j],
                     rms != 0.F ? flux / rms : 0.F, adjacent_noise != 0.F ? flux / adjacent_noise : 0.F,
                     lmad != 0.F ? flux / lmad : 0.F, flux / band.global_mad, band.selfnoise_at_peak[j],
                     band.noise_at_peak[j] / noisemap0_at_peak, band.beam_chi2[j], band.boost[j]);
    }
  }
}

breifast::ProcessPlaneResult assemble_plane_result(const GpuPlaneHostResult &plane,
                                                   const std::vector<GpuBandLabels> &labels, float max_to_std_ratio,
                                                   bool use_rms_noise_estimator,
                                                   std::optional<breifast::SpectralVarianceStats> variance_stats)
{
  breifast::ProcessPlaneResult result;
  assemble_plane_detections(plane, labels, max_to_std_ratio, use_rms_noise_estimator, result.detections);

  result.min_sigma = plane.min_sigma;
  result.max_sigma = plane.max_sigma;
  for (size_t i = 0; i < result.histograms.size(); ++i)
  {
    result.histograms[i].resize(plane.histograms[i].size());
    std::transform(plane.histograms[i].begin(), plane.histograms[i].end(), result.histograms[i].begin(),
                   [](const unsigned long long value) { return static_cast<long>(value); });
  }
  result.median_noises = plane.median_noises;
  result.variance_stats = std::move(variance_stats);
  return result;
}

breifast::ProcessPlaneResult gpu_process_plane_result(
    std::vector<GpuBandJob> &jobs, const std::vector<GpuBandLabels> &labels, float max_to_std_ratio,
    bool use_rms_noise_estimator, std::optional<breifast::SpectralVarianceStats> variance_stats)
{
  GpuPlaneHostResult plane;
  gpu_process_plane(jobs, plane);
  return assemble_plane_result(plane, labels, max_to_std_ratio, use_rms_noise_estimator, std::move(variance_stats));
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

std::vector<float> build_gaussian_beam_kernel_vector(long halfsize, float sigma_major, float sigma_minor, float rotate)
{
  const auto kernel = breifast::build_gaussian_beam_kernel(halfsize, sigma_major, sigma_minor, rotate);
  return std::vector<float>(kernel.begin(), kernel.end());
}

float beam_kernel_sum(const float *kernel, long halfsize)
{
  const auto side = static_cast<std::size_t>(2 * halfsize + 1);
  return xt::sum(xt::adapt(kernel, side * side, xt::no_ownership(), std::array<std::size_t, 2>{side, side}))();
}

ProcessPlaneResult gpu_process_plane_batch(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                           const PlaneConfig &config)
{
  const std::size_t n_y = item.stokes_cube.shape(1);
  const std::size_t n_x = item.stokes_cube.shape(2);
  const std::size_t n_channels = item.stokes_cube.shape(0);

  std::vector<GpuBandSpec> bands;
  bands.reserve(config.band_slices.size());
  for (const auto &band : config.band_slices)
  {
    bands.push_back({band.start, band.stop, band.selfnoise_scale_factor, band.label});
  }

  std::vector<float> fixed_beam_storage;
  if (config.beam_size_enabled && config.fixed_beam_kernel)
  {
    fixed_beam_storage.assign(config.fixed_beam_kernel->begin(), config.fixed_beam_kernel->end());
  }

  const GpuPlaneJobSpec spec{
      .cube = item.stokes_cube.data(),
      .weights = item.weights.data(),
      .nonzero_channels = item.nonzero_channels.data(),
      .accept_mask = group.accept_mask.data(),
      .mean_image = group.mean_image ? group.mean_image->data() : nullptr,
      .selfnoise_offset = group.selfnoise_offset.data(),
      .selfnoise_raw = group.selfnoise_raw.data(),
      .dde_noise = group.dde_noise.data(),
      .snr_cutoff = group.snr_cutoff ? group.snr_cutoff->data() : nullptr,
      .flux_cutoff = group.flux_cutoff ? group.flux_cutoff->data() : nullptr,
      .baseline_flux = config.baseline_flux ? config.baseline_flux->data() : nullptr,
      .beam_major_axis = item.beam_major_axis ? item.beam_major_axis->data() : nullptr,
      .beam_minor_axis = item.beam_minor_axis ? item.beam_minor_axis->data() : nullptr,
      .beam_position_angle = item.beam_position_angle ? item.beam_position_angle->data() : nullptr,
      .fixed_beam_kernel = fixed_beam_storage.empty() ? nullptr : fixed_beam_storage.data(),
      .stokes_label = item.stokes_label,
      .bands = std::move(bands),
      .n_y = n_y,
      .n_x = n_x,
      .n_channels = n_channels,
      .fixed_beam_halfsize = config.fixed_beam_halfsize,
      .boxsize = config.boxsize,
      .inner_boxsize = config.inner_boxsize,
      .histogram_detail = config.histogram_detail,
      .max_sigma_bin = config.max_sigma_bin,
      .ddenoise_scaling_factor = config.ddenoise_scaling_factor,
      .reject_noise_threshold = config.reject_noise_threshold,
      .threshold = config.threshold,
      .max_to_std_ratio_value = config.max_to_std_ratio_value(),
      .use_rms_noise_estimator = config.use_rms_noise_estimator,
      .subtract_mean = config.subtract_mean,
      .beam_size_enabled = config.beam_size_enabled,
  };

  GpuPlaneJobs plane_jobs(spec);

  std::optional<SpectralVarianceStats> variance_stats;
  if (n_channels > 1)
  {
    variance_stats = gpu_compute_spectral_variance_stats(plane_jobs.cube, plane_jobs.nonzero_channels, n_y, n_x,
                                                         n_channels);
  }

  return gpu_process_plane_result(plane_jobs.jobs, plane_jobs.labels, plane_jobs.max_to_std_ratio_value,
                                  plane_jobs.use_rms_noise_estimator, std::move(variance_stats));
}
} // namespace breifast::gpu
