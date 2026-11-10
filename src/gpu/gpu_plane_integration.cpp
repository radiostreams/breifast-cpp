#include "gpu_plane_integration.hpp"

#include <optional>
#include <utility>
#include <vector>

#include "cpu/beam_kernel.hpp"
#include "gpu_plane_jobs.hpp"
#include "gpu_plane_result.hpp"
#include "gpu_spectral_variance.hpp"

namespace breifast::gpu
{
std::vector<float> build_gaussian_beam_kernel_vector(long halfsize, float sigma_major, float sigma_minor, float rotate)
{
  const auto kernel = breifast::build_gaussian_beam_kernel(halfsize, sigma_major, sigma_minor, rotate);
  return std::vector<float>(kernel.begin(), kernel.end());
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
