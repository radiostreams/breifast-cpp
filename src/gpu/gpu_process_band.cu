#include "gpu_process_band.cuh"

#include <algorithm>
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
  CHECK_GPU(cudaMemcpyAsync(&host_flag, flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaStreamSynchronize(stream));
  return host_flag;
}

unsigned int read_count(const unsigned int *count, GpuStream stream)
{
  unsigned int host_count = 0;
  CHECK_GPU(cudaMemcpyAsync(&host_count, count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaStreamSynchronize(stream));
  return host_count;
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
      input.accept_mask == nullptr || input.selfnoise_offset == nullptr || input.selfnoise_raw == nullptr ||
      input.dde_noise == nullptr)
  {
    throw std::runtime_error("gpu_process_band: required input pointer is null");
  }

  require_size("cube", input.sizes.cube, channel_pixels);
  require_size("weights", input.sizes.weights, input.n_channels);
  require_size("nonzero_channels", input.sizes.nonzero_channels, input.n_channels);
  require_size("accept_mask", input.sizes.accept_mask, channel_pixels);
  require_size("selfnoise_offset", input.sizes.selfnoise_offset, input.n_pixels);
  require_size("selfnoise_raw", input.sizes.selfnoise_raw, input.n_pixels);
  require_size("dde_noise", input.sizes.dde_noise, input.n_pixels);
  if (input.mean_image != nullptr)
  {
    require_size("mean_image", input.sizes.mean_image, channel_pixels);
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
  if (input.boxsize < 0 || input.inner_boxsize < 0)
  {
    throw std::runtime_error("gpu_process_band: boxsize and inner_boxsize must be non-negative");
  }

  const long radius = input.boxsize / 2;
  const std::size_t stamp_side = static_cast<std::size_t>(2 * radius + 1);
  const std::size_t stamp_pixels = stamp_side * stamp_side;
  if (stamp_pixels != workspace.sizes.stamp_pixels)
  {
    throw std::runtime_error("gpu_process_band: stamp_pixels mismatch with boxsize");
  }

  std::size_t beam_stamp_pixels = 0;
  if (input.beam_kernel != nullptr)
  {
    if (input.beam_halfsize < 0)
    {
      throw std::runtime_error("gpu_process_band: beam_halfsize must be non-negative");
    }
    const std::size_t beam_side = static_cast<std::size_t>(2 * input.beam_halfsize + 1);
    beam_stamp_pixels = beam_side * beam_side;
    if (beam_stamp_pixels != workspace.sizes.beam_stamp_pixels)
    {
      throw std::runtime_error("gpu_process_band: beam_stamp_pixels mismatch with beam_halfsize");
    }
    if (!(input.max_to_std_ratio_value > 0.F))
    {
      throw std::runtime_error("gpu_process_band: max_to_std_ratio_value must be positive");
    }
  }

  const unsigned int island_capacity =
      static_cast<unsigned int>(std::min(workspace.sizes.stamp_capacity, result.n_pixels));
  if (island_capacity == 0)
  {
    throw std::runtime_error("gpu_process_band: island capacity must be greater than 0");
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
                          input.n_pixels, input.band_start, input.band_stop, input.inv_band_size, stream);

  // Both noise modes need it: the band result reports it, and it is the base noise without an RMS map.
  gpu_median_abs(workspace.plane, input.n_pixels, result.global_mad, 1.4826F, workspace.median_scratch, stream);
  const float *rms = nullptr;
  const float *base_noise = result.global_mad;
  if (input.use_rms_noise_estimator)
  {
    gpu_rms_noise_map(workspace.plane, input.n_y, input.n_x, input.boxsize, workspace.row_sums, workspace.rms_map,
                      stream);
    gpu_median(workspace.rms_map, input.n_pixels, result.rms_floor, workspace.median_scratch, stream);
    rms = workspace.rms_map;
    base_noise = result.rms_floor;
  }
  else
  {
    CHECK_GPU(cudaMemsetAsync(result.rms_floor, 0, sizeof(float), stream)); // 0 in the global MAD mode, as on the CPU
  }
  gpu_combine_noise_map(rms, base_noise, input.selfnoise_offset, input.dde_noise, input.selfnoise_scale_factor,
                        input.ddenoise_scaling_factor, workspace.noise_map, input.n_pixels, stream);

  gpu_median(workspace.noise_map, input.n_pixels, result.median_noise, workspace.median_scratch, stream);

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

  gpu_label_islands(workspace.mask, input.n_y, input.n_x, workspace.labels, workspace.label_count,
                    workspace.label_scratch, stream);

  const unsigned int label_count = read_count(workspace.label_count, stream);
  if (label_count > island_capacity)
  {
    throw std::runtime_error("gpu_process_band: island count " + std::to_string(label_count) +
                             " exceeds island capacity " + std::to_string(island_capacity));
  }
  result.label_count = label_count;

  gpu_label_stats(workspace.labels, workspace.snr, workspace.noise_map, input.n_pixels, input.n_x,
                  workspace.label_count, island_capacity, result.peak_value, result.peak_local_noise,
                  result.peak_y, result.peak_x, result.island_size, workspace.island_scratch, stream);

  gpu_stamp_extract(workspace.plane, input.n_y, input.n_x, result.peak_y, result.peak_x, workspace.label_count,
                    island_capacity, radius, input.inner_boxsize, workspace.donut_stamps, stream);
  gpu_stamp_abs_extreme(workspace.donut_stamps, workspace.label_count, island_capacity, stamp_pixels,
                        result.stamp_extreme, stream);
  gpu_donut_std(workspace.donut_stamps, workspace.label_count, island_capacity, stamp_pixels, result.donut_std, stream);
  gpu_donut_median_abs(workspace.donut_stamps, workspace.label_count, island_capacity, stamp_pixels, 1.4826F,
                       result.donut_lmad, stream);

  if (input.beam_kernel != nullptr)
  {
    gpu_stamp_extract_no_zero(workspace.plane, input.n_y, input.n_x, result.peak_y, result.peak_x,
                              workspace.label_count, island_capacity, input.beam_halfsize, workspace.beam_stamps,
                              stream);
    gpu_stamp_beam_chi2(workspace.beam_stamps, input.beam_kernel, input.beam_kernel_sum, workspace.label_count,
                        island_capacity, beam_stamp_pixels, result.stamp_extreme, 1.F / input.max_to_std_ratio_value,
                        result.beam_chi2, stream);
  }
  else
  {
    CHECK_GPU(cudaMemsetAsync(result.beam_chi2, 0, island_capacity * sizeof(float), stream));
  }

  gpu_stamp_extract_no_zero(workspace.plane, input.n_y, input.n_x, result.peak_y, result.peak_x,
                            workspace.label_count, island_capacity, 0, result.flux, stream);
  gpu_stamp_extract_no_zero(workspace.noise_map, input.n_y, input.n_x, result.peak_y, result.peak_x,
                            workspace.label_count, island_capacity, 0, result.noise_at_peak, stream);
  if (input.use_rms_noise_estimator)
  {
    gpu_stamp_extract_no_zero(workspace.rms_map, input.n_y, input.n_x, result.peak_y, result.peak_x,
                              workspace.label_count, island_capacity, 0, result.rms_at_peak, stream);
  }
  else
  {
    CHECK_GPU(cudaMemsetAsync(result.rms_at_peak, 0, island_capacity * sizeof(float), stream));
  }
  gpu_stamp_extract_no_zero(input.selfnoise_raw, input.n_y, input.n_x, result.peak_y, result.peak_x,
                            workspace.label_count, island_capacity, 0, result.selfnoise_at_peak, stream);

  if (input.baseline_flux != nullptr)
  {
    gpu_island_boost(workspace.labels, input.baseline_flux, input.n_pixels, workspace.label_count, island_capacity,
                     result.flux, result.boost, result.boost_found, workspace.island_scratch, stream);
  }
  else
  {
    CHECK_GPU(cudaMemsetAsync(result.boost, 0, island_capacity * sizeof(float), stream));
    CHECK_GPU(cudaMemsetAsync(result.boost_found, 0, island_capacity * sizeof(bool), stream));
  }

  return GpuBandStatus::detections;
}

GpuBandJob::GpuBandJob(GpuBandInput input, const GpuBandWorkspaceSizes &workspace_sizes,
                       std::size_t result_n_pixels, std::size_t result_histogram_bins)
    : input(input), workspace_sizes(workspace_sizes), result_n_pixels(result_n_pixels),
      result_histogram_bins(result_histogram_bins)
{
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
            job.workspace.emplace(job.workspace_sizes);
            job.result.emplace(job.result_n_pixels, job.result_histogram_bins);
            job.status = gpu_process_band(job.input, *job.workspace, *job.result, job.stream.get());
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
} // namespace breifast::gpu
