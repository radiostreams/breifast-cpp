#include "gpu_plane_integration.hpp"

#include <optional>
#include <utility>
#include <vector>

#include "gpu_plane_jobs.hpp"
#include "gpu_plane_result.hpp"
#include "gpu_spectral_variance.hpp"

namespace breifast::gpu
{
ProcessPlaneResult gpu_process_plane_batch(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                           const PlaneConfig &config)
{
  const std::size_t n_y = item.stokes_cube.shape(1);
  const std::size_t n_x = item.stokes_cube.shape(2);
  const std::size_t n_channels = item.stokes_cube.shape(0);

  GpuPlaneJobSpec spec;
  spec.cube = item.stokes_cube.data();
  spec.weights = item.weights.data();
  spec.nonzero_channels = item.nonzero_channels.data();
  spec.accept_mask = group.accept_mask.data();
  spec.mean_image = group.mean_image ? group.mean_image->data() : nullptr;
  spec.selfnoise_offset = group.selfnoise_offset.data();
  spec.selfnoise_raw = group.selfnoise_raw.data();
  spec.dde_noise = group.dde_noise.data();
  spec.snr_cutoff = group.snr_cutoff ? group.snr_cutoff->data() : nullptr;
  spec.flux_cutoff = group.flux_cutoff ? group.flux_cutoff->data() : nullptr;
  spec.baseline_flux = config.baseline_flux ? config.baseline_flux->data() : nullptr;
  spec.beam_major_axis = item.beam_major_axis ? item.beam_major_axis->data() : nullptr;
  spec.beam_minor_axis = item.beam_minor_axis ? item.beam_minor_axis->data() : nullptr;
  spec.beam_position_angle = item.beam_position_angle ? item.beam_position_angle->data() : nullptr;

  std::vector<float> fixed_beam_storage;
  if (config.beam_size_enabled && config.fixed_beam_kernel)
  {
    fixed_beam_storage.assign(config.fixed_beam_kernel->begin(), config.fixed_beam_kernel->end());
    spec.fixed_beam_kernel = fixed_beam_storage.data();
    spec.fixed_beam_halfsize = config.fixed_beam_halfsize;
  }

  spec.stokes_label = item.stokes_label;
  spec.bands.reserve(config.band_slices.size());
  for (const auto &band : config.band_slices)
  {
    spec.bands.push_back({band.start, band.stop, band.selfnoise_scale_factor, band.label});
  }
  spec.n_y = n_y;
  spec.n_x = n_x;
  spec.n_channels = n_channels;
  spec.boxsize = config.boxsize;
  spec.inner_boxsize = config.inner_boxsize;
  spec.histogram_detail = config.histogram_detail;
  spec.max_sigma_bin = config.max_sigma_bin;
  spec.ddenoise_scaling_factor = config.ddenoise_scaling_factor;
  spec.reject_noise_threshold = config.reject_noise_threshold;
  spec.threshold = config.threshold;
  spec.max_to_std_ratio_value = config.max_to_std_ratio_value();
  spec.use_rms_noise_estimator = config.use_rms_noise_estimator;
  spec.subtract_mean = config.subtract_mean;
  spec.beam_size_enabled = config.beam_size_enabled;

  GpuPlaneJobs plane_jobs = build_plane_jobs(spec);
  try
  {
    GpuPlaneHostResult host;
    gpu_process_plane(plane_jobs.jobs, host);

    std::optional<SpectralVarianceStats> variance_stats;
    if (n_channels > 1)
    {
      variance_stats = gpu_compute_spectral_variance_stats(plane_jobs.cube, plane_jobs.nonzero_channels, n_y, n_x,
                                                           n_channels);
    }

    ProcessPlaneResult result = gpu_process_plane_result(plane_jobs.jobs, plane_jobs.labels,
                                                         plane_jobs.max_to_std_ratio_value,
                                                         plane_jobs.use_rms_noise_estimator,
                                                         std::move(variance_stats));
    destroy_plane_jobs(plane_jobs);
    return result;
  }
  catch (...)
  {
    destroy_plane_jobs(plane_jobs);
    throw;
  }
}
} // namespace breifast::gpu
