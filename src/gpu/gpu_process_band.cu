#include "gpu_process_band.cuh"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

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

} // namespace

bool gpu_process_band(const GpuPlaneInput &plane, const GpuBand &band, GpuBandWorkspace &workspace,
                      GpuBandResult &result, GpuStream stream)
{
  const std::size_t n_pixels = plane.n_pixels();
  assert(n_pixels > 0 && plane.n_channels > 0);
  assert(band.start >= 0 && band.start < band.stop && static_cast<std::size_t>(band.stop) <= plane.n_channels);
  assert(plane.max_sigma_bin > 0);
  assert(plane.boxsize >= 0 && plane.inner_boxsize >= 0);
  assert(plane.cube != nullptr && plane.weights != nullptr && plane.accept_mask != nullptr &&
         plane.selfnoise_offset != nullptr && plane.selfnoise_raw != nullptr && plane.dde_noise != nullptr);
  assert(band.beam_kernel == nullptr || (band.beam_halfsize >= 0 && plane.max_to_std_ratio > 0.F));

  const long radius = plane.boxsize / 2;
  const auto stamp_side = static_cast<std::size_t>(2 * radius + 1);
  const std::size_t stamp_pixels = stamp_side * stamp_side;
  const auto beam_side = static_cast<std::size_t>(2 * band.beam_halfsize + 1);
  const std::size_t beam_stamp_pixels = beam_side * beam_side;
  const float inv_band_size = 1.F / static_cast<float>(band.stop - band.start);
  const auto island_capacity = static_cast<unsigned int>(std::min(n_pixels, max_stamp_capacity));
  workspace.ensure(plane, band);
  result.ensure(island_capacity, static_cast<std::size_t>(2 * plane.max_sigma_bin));

  gpu_mask_or(plane.accept_mask, n_pixels, band.start, band.stop, workspace.mask, workspace.any_flag, stream);
  if (read_flag(workspace.any_flag, stream) == 0)
  {
    return false;
  }

  gpu_weighted_band_image(plane.cube, plane.weights, plane.mean_image, workspace.plane, n_pixels, band.start, band.stop,
                          inv_band_size, stream);

  // Both noise modes need it: the band result reports it, and it is the base noise without an RMS map.
  gpu_median_abs(workspace.plane, n_pixels, result.global_mad, 1.4826F, workspace.median_scratch, stream);
  const float *rms = nullptr;
  const float *base_noise = result.global_mad;
  if (plane.use_rms_noise_estimator)
  {
    gpu_rms_noise_map(workspace.plane, plane.n_y, plane.n_x, plane.boxsize, workspace.row_sums, workspace.rms_map,
                      stream);
    gpu_median(workspace.rms_map, n_pixels, result.rms_floor, workspace.median_scratch, stream);
    rms = workspace.rms_map;
    base_noise = result.rms_floor;
  }
  else
  {
    CHECK_GPU(cudaMemsetAsync(result.rms_floor, 0, sizeof(float), stream)); // 0 in the global MAD mode, as on the CPU
  }
  gpu_combine_noise_map(rms, base_noise, plane.selfnoise_offset, plane.dde_noise, band.selfnoise_scale_factor,
                        plane.ddenoise_scaling_factor, workspace.noise_map, n_pixels, stream);

  gpu_median(workspace.noise_map, n_pixels, result.median_noise, workspace.median_scratch, stream);

  common::SnrFilterParams params;
  params.inv_band_size = inv_band_size;
  params.band_start = band.start;
  params.band_stop = band.stop;
  params.max_sigma_bin = plane.max_sigma_bin;
  params.histogram_detail = plane.histogram_detail;
  params.reject_noise_threshold = plane.reject_noise_threshold;
  params.threshold = plane.threshold;

  gpu_snr_and_filters(workspace.plane, workspace.noise_map, plane.snr_cutoff, plane.flux_cutoff, result.median_noise,
                      n_pixels, params, workspace.snr, workspace.mask, result.histogram0, result.histogram1,
                      result.histogram2, result.min_sigma, result.max_sigma, stream);

  gpu_any_true(workspace.mask, n_pixels, workspace.any_flag, stream);
  if (read_flag(workspace.any_flag, stream) == 0)
  {
    result.label_count = 0;
    return true;
  }

  gpu_label_islands(workspace.mask, plane.n_y, plane.n_x, workspace.labels, workspace.label_count,
                    workspace.label_scratch, stream);

  const unsigned int label_count = read_count(workspace.label_count, stream);
  if (label_count > island_capacity)
  {
    throw std::runtime_error("gpu_process_band: island count " + std::to_string(label_count) +
                             " exceeds island capacity " + std::to_string(island_capacity));
  }
  result.label_count = label_count;

  gpu_label_stats(workspace.labels, workspace.snr, workspace.noise_map, n_pixels, plane.n_x, workspace.label_count,
                  island_capacity, result.islands.peak_value, result.islands.peak_local_noise, result.islands.peak_y,
                  result.islands.peak_x, result.islands.island_size, workspace.island_scratch, stream);

  gpu_stamp_extract(workspace.plane, plane.n_y, plane.n_x, result.islands.peak_y, result.islands.peak_x,
                    workspace.label_count, island_capacity, radius, plane.inner_boxsize, workspace.donut_stamps,
                    stream);
  gpu_stamp_abs_extreme(workspace.donut_stamps, workspace.label_count, island_capacity, stamp_pixels,
                        result.islands.stamp_extreme, stream);
  gpu_donut_std(workspace.donut_stamps, workspace.label_count, island_capacity, stamp_pixels, result.islands.donut_std,
                stream);
  gpu_donut_median_abs(workspace.donut_stamps, workspace.label_count, island_capacity, stamp_pixels, 1.4826F,
                       result.islands.donut_lmad, stream);

  if (band.beam_kernel != nullptr)
  {
    gpu_stamp_extract_no_zero(workspace.plane, plane.n_y, plane.n_x, result.islands.peak_y, result.islands.peak_x,
                              workspace.label_count, island_capacity, band.beam_halfsize, workspace.beam_stamps,
                              stream);
    gpu_stamp_beam_chi2(workspace.beam_stamps, band.beam_kernel, band.beam_kernel_sum, workspace.label_count,
                        island_capacity, beam_stamp_pixels, result.islands.stamp_extreme, 1.F / plane.max_to_std_ratio,
                        result.islands.beam_chi2, stream);
  }
  else
  {
    CHECK_GPU(cudaMemsetAsync(result.islands.beam_chi2, 0, island_capacity * sizeof(float), stream));
  }

  gpu_stamp_extract_no_zero(workspace.plane, plane.n_y, plane.n_x, result.islands.peak_y, result.islands.peak_x,
                            workspace.label_count, island_capacity, 0, result.islands.flux, stream);
  gpu_stamp_extract_no_zero(workspace.noise_map, plane.n_y, plane.n_x, result.islands.peak_y, result.islands.peak_x,
                            workspace.label_count, island_capacity, 0, result.islands.noise_at_peak, stream);
  if (plane.use_rms_noise_estimator)
  {
    gpu_stamp_extract_no_zero(workspace.rms_map, plane.n_y, plane.n_x, result.islands.peak_y, result.islands.peak_x,
                              workspace.label_count, island_capacity, 0, result.islands.rms_at_peak, stream);
  }
  else
  {
    CHECK_GPU(cudaMemsetAsync(result.islands.rms_at_peak, 0, island_capacity * sizeof(float), stream));
  }
  gpu_stamp_extract_no_zero(plane.selfnoise_raw, plane.n_y, plane.n_x, result.islands.peak_y, result.islands.peak_x,
                            workspace.label_count, island_capacity, 0, result.islands.selfnoise_at_peak, stream);

  if (plane.baseline_flux != nullptr)
  {
    gpu_island_boost(workspace.labels, plane.baseline_flux, n_pixels, workspace.label_count, island_capacity,
                     result.islands.flux, result.islands.boost, workspace.island_scratch, stream);
  }
  else
  {
    CHECK_GPU(cudaMemsetAsync(result.islands.boost, 0, island_capacity * sizeof(float), stream));
  }

  return true;
}
} // namespace breifast::gpu
