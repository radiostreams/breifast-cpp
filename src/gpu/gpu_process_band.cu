#include "gpu_process_band.cuh"

#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
namespace
{
int read_flag(int *flag, GpuStream stream)
{
  int host_flag = 0;
  gpuMemcpyDeviceToHostAsync(&host_flag, flag, sizeof(int), stream);
  gpuStreamSynchronize(stream);
  return host_flag;
}

void require_size(const char *name, const std::size_t actual, const std::size_t required)
{
  if (actual < required)
  {
    throw std::runtime_error(std::string("gpu_process_band: ") + name + " is too small");
  }
}

} // namespace

GpuBandStatus gpu_process_band(const GpuBandInput &input, GpuBandWorkspace &workspace, GpuBandResult &result,
                               GpuStream stream)
{
  if (input.n_y == 0 || input.n_x == 0)
  {
    throw std::runtime_error("gpu_process_band: n_y and n_x must be greater than 0");
  }
  if (input.n_y > std::numeric_limits<std::size_t>::max() / input.n_x)
  {
    throw std::runtime_error("gpu_process_band: n_y * n_x overflows");
  }
  if (input.n_y * input.n_x != input.n_pixels)
  {
    throw std::runtime_error("gpu_process_band: n_y * n_x must equal n_pixels");
  }
  if (input.n_channels > std::numeric_limits<std::size_t>::max() / input.n_pixels)
  {
    throw std::runtime_error("gpu_process_band: n_channels * n_pixels overflows");
  }
  const std::size_t channel_pixels = input.n_channels * input.n_pixels;
  if (input.n_channels == 0)
  {
    throw std::runtime_error("gpu_process_band: n_channels must be greater than 0");
  }
  if (input.band_start < 0 || input.band_stop <= input.band_start ||
      static_cast<std::size_t>(input.band_stop) > input.n_channels)
  {
    throw std::runtime_error("gpu_process_band: invalid band range");
  }
  if (input.max_sigma_bin <= 0)
  {
    throw std::runtime_error("gpu_process_band: max_sigma_bin must be positive");
  }
  if (input.cube == nullptr || input.weights == nullptr || input.nonzero_channels == nullptr ||
      input.accept_mask == nullptr || input.selfnoise_offset == nullptr || input.dde_noise == nullptr)
  {
    throw std::runtime_error("gpu_process_band: required input pointer is null");
  }

  require_size("cube", input.sizes.cube, channel_pixels);
  require_size("weights", input.sizes.weights, input.n_channels);
  require_size("nonzero_channels", input.sizes.nonzero_channels, input.n_channels);
  require_size("accept_mask", input.sizes.accept_mask, channel_pixels);
  require_size("selfnoise_offset", input.sizes.selfnoise_offset, input.n_pixels);
  require_size("dde_noise", input.sizes.dde_noise, input.n_pixels);
  if (input.mean_image != nullptr)
  {
    require_size("mean_image", input.sizes.mean_image, channel_pixels);
  }
  if (input.selfnoise_raw != nullptr)
  {
    require_size("selfnoise_raw", input.sizes.selfnoise_raw, input.n_pixels);
  }
  if (input.snr_cutoff != nullptr)
  {
    require_size("snr_cutoff", input.sizes.snr_cutoff, channel_pixels);
  }
  if (input.flux_cutoff != nullptr)
  {
    require_size("flux_cutoff", input.sizes.flux_cutoff, channel_pixels);
  }
  if (input.baseline_flux != nullptr)
  {
    require_size("baseline_flux", input.sizes.baseline_flux, input.n_pixels);
  }
  if (input.beam_kernel != nullptr)
  {
    require_size("beam_kernel", input.sizes.beam_kernel, workspace.sizes.beam_stamp_pixels);
  }
  if (workspace.sizes.n_pixels != input.n_pixels || result.n_pixels != input.n_pixels)
  {
    throw std::runtime_error("gpu_process_band: workspace/result pixel count mismatch");
  }
  if (result.histogram_bins != static_cast<std::size_t>(2 * input.max_sigma_bin))
  {
    throw std::runtime_error("gpu_process_band: histogram_bins must equal 2 * max_sigma_bin");
  }

  const std::size_t band_size = static_cast<std::size_t>(input.band_stop - input.band_start);
  gpu_any_true(input.nonzero_channels + input.band_start, band_size, workspace.any_flag, stream);
  if (read_flag(workspace.any_flag, stream) == 0)
  {
    return GpuBandStatus::skipped;
  }

  gpu_mask_or(input.accept_mask, input.n_pixels, input.band_start, input.band_stop, workspace.mask,
              workspace.any_flag, stream);
  if (read_flag(workspace.any_flag, stream) == 0)
  {
    return GpuBandStatus::skipped;
  }

  gpu_weighted_band_image(input.cube, input.weights, input.subtract_mean ? input.mean_image : nullptr, workspace.plane,
                          input.n_pixels, input.band_start, input.band_stop, input.inv_weights_sum,
                          input.inv_band_size, stream);

  if (input.use_rms_noise_estimator)
  {
    gpu_rms_noise_map(workspace.plane, input.n_y, input.n_x, input.boxsize, workspace.row_sums, workspace.rms_map,
                      stream);
    gpu_median(workspace.rms_map, input.n_pixels, result.rms_floor, stream);
    gpu_combine_noise_map_rms(workspace.rms_map, result.rms_floor, input.selfnoise_offset, input.dde_noise,
                              input.selfnoise_scale_factor, input.ddenoise_scaling_factor, workspace.noise_map,
                              input.n_pixels, stream);
    gpu_median_abs(workspace.plane, input.n_pixels, result.global_mad, 1.4826F, stream);
  }
  else
  {
    gpu_median_abs(workspace.plane, input.n_pixels, result.global_mad, 1.4826F, stream);
    gpu_combine_noise_map_constant(result.global_mad, input.selfnoise_offset, input.dde_noise,
                                   input.selfnoise_scale_factor,
                                   input.ddenoise_scaling_factor, workspace.noise_map, input.n_pixels, stream);
  }

  gpu_median(workspace.noise_map, input.n_pixels, result.median_noise, stream);

  common::SnrFilterParams params;
  params.inv_band_size = input.inv_band_size;
  params.band_start = input.band_start;
  params.band_stop = input.band_stop;
  params.max_sigma_bin = input.max_sigma_bin;
  params.histogram_detail = input.histogram_detail;
  params.reject_noise_threshold = input.reject_noise_threshold;
  params.threshold = input.threshold;

  gpu_snr_and_filters(workspace.plane, workspace.noise_map, input.snr_cutoff, input.flux_cutoff, result.median_noise,
                      input.n_pixels, params, workspace.snr, workspace.mask, result.histogram0, result.histogram1,
                      result.histogram2, result.min_sigma, result.max_sigma, stream);

  gpu_any_true(workspace.mask, input.n_pixels, workspace.any_flag, stream);
  if (read_flag(workspace.any_flag, stream) == 0)
  {
    return GpuBandStatus::no_detections;
  }

  throw std::runtime_error("gpu_process_band: detection stage not implemented");
}

void gpu_process_bands(std::vector<GpuBandJob> &jobs)
{
  std::vector<std::thread> threads;
  threads.reserve(jobs.size());

  for (auto &job : jobs)
  {
    threads.emplace_back(
        [&job]()
        {
          try
          {
            cudaStream_t raw_stream{};
            gpuStreamCreate(&raw_stream);
            job.stream = GpuStream{raw_stream};
            job.workspace = create_band_workspace(job.workspace_sizes, job.stream);
            job.result = create_band_result(job.result_n_pixels, job.result_histogram_bins, job.stream);
            job.status = gpu_process_band(job.input, job.workspace, job.result, job.stream);
          }
          catch (...)
          {
            job.error = std::current_exception();
          }
        });
  }

  for (auto &thread : threads)
  {
    thread.join();
  }
}

void destroy_band_job(GpuBandJob &job)
{
  if (job.stream.handle_ != nullptr)
  {
    destroy_band_result(job.result, job.stream);
    destroy_band_workspace(job.workspace, job.stream);
    gpuStreamSynchronize(job.stream);
    gpuStreamDestroy(job.stream);
  }
  job = GpuBandJob{};
}
} // namespace breifast::gpu
