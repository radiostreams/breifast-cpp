#include "gpu_plane_jobs.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "cpu/beam_kernel.hpp"

namespace breifast::gpu
{
namespace
{
std::size_t checked_pixels(const std::size_t n_y, const std::size_t n_x)
{
  if (n_y == 0 || n_x == 0)
  {
    throw std::runtime_error("build_plane_jobs: zero plane dimension");
  }
  if (n_y > std::numeric_limits<std::size_t>::max() / n_x)
  {
    throw std::runtime_error("build_plane_jobs: pixel count overflow");
  }
  return n_y * n_x;
}

std::size_t checked_channels(const std::size_t n_channels, const std::size_t n_pixels)
{
  if (n_channels == 0)
  {
    throw std::runtime_error("build_plane_jobs: zero channel count");
  }
  if (n_pixels > std::numeric_limits<std::size_t>::max() / n_channels)
  {
    throw std::runtime_error("build_plane_jobs: channel/pixel count overflow");
  }
  return n_channels * n_pixels;
}

void require_pointer(const void *pointer, const char *name)
{
  if (pointer == nullptr)
  {
    throw std::runtime_error(std::string("build_plane_jobs: missing ") + name);
  }
}
} // namespace

GpuPlaneJobs build_plane_jobs(const GpuPlaneJobSpec &spec)
{
  const std::size_t n_pixels = checked_pixels(spec.n_y, spec.n_x);
  const std::size_t channel_pixels = checked_channels(spec.n_channels, n_pixels);
  if (spec.max_sigma_bin <= 0)
  {
    throw std::runtime_error("build_plane_jobs: max_sigma_bin must be positive");
  }
  if (spec.boxsize < 0 || spec.inner_boxsize < 0)
  {
    throw std::runtime_error("build_plane_jobs: boxsize and inner_boxsize must be non-negative");
  }
  if (spec.beam_size_enabled && !(spec.max_to_std_ratio_value > 0.F))
  {
    throw std::runtime_error("build_plane_jobs: max_to_std_ratio_value must be positive");
  }
  require_pointer(spec.cube, "cube");
  require_pointer(spec.weights, "weights");
  require_pointer(spec.nonzero_channels, "nonzero_channels");
  require_pointer(spec.accept_mask, "accept_mask");
  require_pointer(spec.selfnoise_offset, "selfnoise_offset");
  require_pointer(spec.selfnoise_raw, "selfnoise_raw");
  require_pointer(spec.dde_noise, "dde_noise");

  GpuPlaneJobs result;
  result.n_y = spec.n_y;
  result.n_x = spec.n_x;
  result.n_channels = spec.n_channels;
  result.max_to_std_ratio_value = spec.max_to_std_ratio_value;
  result.use_rms_noise_estimator = spec.use_rms_noise_estimator;

  const auto upload = [&result](const void *host, const std::size_t bytes)
  {
    void *device = gpu_upload_buffer(host, bytes);
    if (device != nullptr)
    {
      result.buffers.push_back(device);
    }
    return device;
  };

  try
  {
    result.cube = static_cast<const float *>(upload(spec.cube, channel_pixels * sizeof(float)));
    const auto *weights = static_cast<const float *>(upload(spec.weights, spec.n_channels * sizeof(float)));
    result.nonzero_channels =
        static_cast<const bool *>(upload(spec.nonzero_channels, spec.n_channels * sizeof(bool)));
    const auto *accept_mask =
        static_cast<const bool *>(upload(spec.accept_mask, channel_pixels * sizeof(bool)));
    const auto *selfnoise_offset =
        static_cast<const float *>(upload(spec.selfnoise_offset, n_pixels * sizeof(float)));
    const auto *selfnoise_raw = static_cast<const float *>(upload(spec.selfnoise_raw, n_pixels * sizeof(float)));
    const auto *dde_noise = static_cast<const float *>(upload(spec.dde_noise, n_pixels * sizeof(float)));

    const auto *mean_image =
        (spec.subtract_mean && spec.mean_image != nullptr)
            ? static_cast<const float *>(upload(spec.mean_image, channel_pixels * sizeof(float)))
            : nullptr;
    const auto *snr_cutoff = spec.snr_cutoff != nullptr
                                 ? static_cast<const float *>(upload(spec.snr_cutoff, channel_pixels * sizeof(float)))
                                 : nullptr;
    const auto *flux_cutoff = spec.flux_cutoff != nullptr
                                  ? static_cast<const float *>(upload(spec.flux_cutoff, channel_pixels * sizeof(float)))
                                  : nullptr;
    const auto *baseline_flux = spec.baseline_flux != nullptr
                                    ? static_cast<const float *>(upload(spec.baseline_flux, n_pixels * sizeof(float)))
                                    : nullptr;

    const long radius = spec.boxsize / 2;
    const std::size_t stamp_side = static_cast<std::size_t>(2 * radius + 1);
    const std::size_t stamp_pixels = stamp_side * stamp_side;
    const std::size_t histogram_bins = static_cast<std::size_t>(2 * spec.max_sigma_bin);

    const bool use_fixed_beam = spec.beam_size_enabled && spec.fixed_beam_kernel != nullptr;
    if (use_fixed_beam && spec.fixed_beam_halfsize < 0)
    {
      throw std::runtime_error("build_plane_jobs: fixed_beam_halfsize must be non-negative");
    }
    const float *fixed_beam_device = nullptr;
    if (use_fixed_beam)
    {
      const std::size_t beam_side = static_cast<std::size_t>(2 * spec.fixed_beam_halfsize + 1);
      fixed_beam_device = static_cast<const float *>(
          upload(spec.fixed_beam_kernel, beam_side * beam_side * sizeof(float)));
    }

    result.jobs.reserve(spec.bands.size());
    result.labels.reserve(spec.bands.size());

    for (const auto &band : spec.bands)
    {
      if (band.start < 0 || band.stop <= band.start ||
          static_cast<std::size_t>(band.stop) > spec.n_channels)
      {
        throw std::runtime_error("build_plane_jobs: invalid band range");
      }

      const std::size_t band_size = static_cast<std::size_t>(band.stop - band.start);
      const float weights_sum = std::accumulate(spec.weights + band.start, spec.weights + band.stop, 0.F);
      const float inv_weights_sum = 1.F / weights_sum;
      const float inv_band_size = 1.F / static_cast<float>(band_size);

      const float *beam_device = nullptr;
      long beam_halfsize = 0;
      std::size_t beam_stamp_pixels = 0;
      if (spec.beam_size_enabled)
      {
        if (use_fixed_beam)
        {
          beam_device = fixed_beam_device;
          beam_halfsize = spec.fixed_beam_halfsize;
        }
        else
        {
          require_pointer(spec.beam_major_axis, "beam_major_axis");
          require_pointer(spec.beam_minor_axis, "beam_minor_axis");
          require_pointer(spec.beam_position_angle, "beam_position_angle");

          float halfsize_accumulator = 0.F;
          for (long i_channel = band.start; i_channel < band.stop; ++i_channel)
          {
            halfsize_accumulator += spec.beam_major_axis[i_channel];
          }
          halfsize_accumulator *= 3.F / static_cast<float>(band_size);
          beam_halfsize = static_cast<long>(halfsize_accumulator) + 1;
          if (beam_halfsize < 0)
          {
            throw std::runtime_error("build_plane_jobs: negative Gaussian beam halfsize");
          }

          float major_weighted = 0.F;
          float minor_weighted = 0.F;
          float angle_weighted = 0.F;
          for (long i_channel = band.start; i_channel < band.stop; ++i_channel)
          {
            const float weight = spec.weights[i_channel];
            major_weighted += spec.beam_major_axis[i_channel] * weight;
            minor_weighted += spec.beam_minor_axis[i_channel] * weight;
            angle_weighted += spec.beam_position_angle[i_channel] * weight;
          }
          major_weighted /= weights_sum;
          minor_weighted /= weights_sum;
          angle_weighted /= weights_sum;

          const auto host_kernel =
              build_gaussian_beam_kernel(beam_halfsize, major_weighted, minor_weighted, angle_weighted);
          beam_device = static_cast<const float *>(upload(host_kernel.data(), host_kernel.size() * sizeof(float)));
        }
        const std::size_t beam_side = static_cast<std::size_t>(2 * beam_halfsize + 1);
        beam_stamp_pixels = beam_side * beam_side;
      }

      GpuBandWorkspaceSizes workspace_sizes;
      workspace_sizes.n_pixels = n_pixels;
      workspace_sizes.stamp_capacity = std::min(n_pixels, max_stamp_capacity);
      workspace_sizes.stamp_pixels = stamp_pixels;
      workspace_sizes.beam_stamp_pixels = beam_stamp_pixels;
      workspace_sizes.histogram_bins = histogram_bins;

      GpuBandJob job;
      job.input.sizes.cube = channel_pixels;
      job.input.sizes.weights = spec.n_channels;
      job.input.sizes.nonzero_channels = spec.n_channels;
      job.input.sizes.accept_mask = channel_pixels;
      job.input.sizes.selfnoise_offset = n_pixels;
      job.input.sizes.selfnoise_raw = n_pixels;
      job.input.sizes.dde_noise = n_pixels;
      job.input.sizes.mean_image = mean_image != nullptr ? channel_pixels : 0;
      job.input.sizes.snr_cutoff = snr_cutoff != nullptr ? channel_pixels : 0;
      job.input.sizes.flux_cutoff = flux_cutoff != nullptr ? channel_pixels : 0;
      job.input.sizes.baseline_flux = baseline_flux != nullptr ? n_pixels : 0;
      job.input.sizes.beam_kernel = beam_stamp_pixels;

      job.input.cube = result.cube;
      job.input.weights = weights;
      job.input.nonzero_channels = result.nonzero_channels;
      job.input.accept_mask = accept_mask;
      job.input.mean_image = mean_image;
      job.input.selfnoise_offset = selfnoise_offset;
      job.input.selfnoise_raw = selfnoise_raw;
      job.input.dde_noise = dde_noise;
      job.input.snr_cutoff = snr_cutoff;
      job.input.flux_cutoff = flux_cutoff;
      job.input.baseline_flux = baseline_flux;
      job.input.beam_kernel = beam_device;

      job.input.n_y = spec.n_y;
      job.input.n_x = spec.n_x;
      job.input.n_pixels = n_pixels;
      job.input.n_channels = spec.n_channels;
      job.input.band_start = band.start;
      job.input.band_stop = band.stop;
      job.input.boxsize = spec.boxsize;
      job.input.inner_boxsize = spec.inner_boxsize;
      job.input.beam_halfsize = beam_halfsize;
      job.input.histogram_detail = spec.histogram_detail;
      job.input.max_sigma_bin = spec.max_sigma_bin;
      job.input.selfnoise_scale_factor = band.selfnoise_scale_factor;
      job.input.inv_band_size = inv_band_size;
      job.input.inv_weights_sum = inv_weights_sum;
      job.input.ddenoise_scaling_factor = spec.ddenoise_scaling_factor;
      job.input.reject_noise_threshold = spec.reject_noise_threshold;
      job.input.threshold = spec.threshold;
      job.input.max_to_std_ratio_value = spec.max_to_std_ratio_value;
      job.input.use_rms_noise_estimator = spec.use_rms_noise_estimator;
      job.input.subtract_mean = mean_image != nullptr;

      job.workspace_sizes = workspace_sizes;
      job.result_n_pixels = n_pixels;
      job.result_histogram_bins = histogram_bins;

      result.jobs.push_back(std::move(job));
      result.labels.push_back({spec.stokes_label, band.label});
    }
  }
  catch (...)
  {
    destroy_plane_jobs(result);
    throw;
  }

  return result;
}

void destroy_plane_jobs(GpuPlaneJobs &plane_jobs) noexcept
{
  for (auto &job : plane_jobs.jobs)
  {
    destroy_band_job(job);
  }
  for (void *buffer : plane_jobs.buffers)
  {
    gpu_free_buffer(buffer);
  }
  plane_jobs = GpuPlaneJobs{};
}
} // namespace breifast::gpu
