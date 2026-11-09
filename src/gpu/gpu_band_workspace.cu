#include "gpu_band_workspace.cuh"

#include <stdexcept>
#include <string>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
namespace
{
template <typename T> void free_buffer(T *&buffer, GpuStream stream)
{
  if (buffer != nullptr)
  {
    gpuFreeAsync(buffer, stream);
    buffer = nullptr;
  }
}
} // namespace

GpuBandWorkspace create_band_workspace(const GpuBandWorkspaceSizes &sizes, GpuStream stream)
{
  validate_band_workspace_sizes(sizes);

  GpuBandWorkspace workspace;
  workspace.sizes = sizes;

  try
  {
    const std::size_t n_pixels = sizes.n_pixels;
    gpuMallocAsync(&workspace.plane, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&workspace.noise_map, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&workspace.snr, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&workspace.mask, n_pixels * sizeof(bool), stream);
    gpuMallocAsync(&workspace.row_sums, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&workspace.rms_map, n_pixels * sizeof(float), stream);

    gpuMallocAsync(&workspace.labels, n_pixels * sizeof(unsigned int), stream);
    gpuMallocAsync(&workspace.label_scratch, n_pixels * sizeof(unsigned int), stream);

    gpuMallocAsync(&workspace.donut_stamps, sizes.stamp_capacity * sizes.stamp_pixels * sizeof(float), stream);
    if (sizes.beam_stamp_pixels > 0)
    {
      gpuMallocAsync(&workspace.beam_stamps, sizes.stamp_capacity * sizes.beam_stamp_pixels * sizeof(float), stream);
    }

    gpuMallocAsync(&workspace.any_flag, sizeof(int), stream);
    gpuMallocAsync(&workspace.label_count, sizeof(unsigned int), stream);
  }
  catch (...)
  {
    destroy_band_workspace(workspace, stream);
    throw;
  }

  return workspace;
}

void destroy_band_workspace(GpuBandWorkspace &workspace, GpuStream stream)
{
  free_buffer(workspace.plane, stream);
  free_buffer(workspace.noise_map, stream);
  free_buffer(workspace.snr, stream);
  free_buffer(workspace.mask, stream);
  free_buffer(workspace.row_sums, stream);
  free_buffer(workspace.rms_map, stream);
  free_buffer(workspace.labels, stream);
  free_buffer(workspace.label_scratch, stream);
  free_buffer(workspace.donut_stamps, stream);
  free_buffer(workspace.beam_stamps, stream);
  free_buffer(workspace.any_flag, stream);
  free_buffer(workspace.label_count, stream);
  workspace = GpuBandWorkspace{};
}

GpuBandResult create_band_result(const std::size_t n_pixels, const std::size_t histogram_bins, GpuStream stream)
{
  if (n_pixels == 0)
  {
    throw std::runtime_error("GpuBandResult: n_pixels must be greater than 0");
  }
  if (histogram_bins == 0)
  {
    throw std::runtime_error("GpuBandResult: histogram_bins must be greater than 0");
  }

  GpuBandResult result;
  result.n_pixels = n_pixels;
  result.histogram_bins = histogram_bins;

  try
  {
    gpuMallocAsync(&result.peak_value, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.peak_local_noise, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.peak_y, n_pixels * sizeof(long), stream);
    gpuMallocAsync(&result.peak_x, n_pixels * sizeof(long), stream);
    gpuMallocAsync(&result.island_size, n_pixels * sizeof(unsigned int), stream);

    gpuMallocAsync(&result.stamp_extreme, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.donut_std, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.donut_lmad, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.beam_chi2, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.flux, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.boost, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.boost_found, n_pixels * sizeof(bool), stream);

    gpuMallocAsync(&result.noise_at_peak, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.rms_at_peak, n_pixels * sizeof(float), stream);
    gpuMallocAsync(&result.selfnoise_at_peak, n_pixels * sizeof(float), stream);

    gpuMallocAsync(&result.histogram0, histogram_bins * sizeof(unsigned long long), stream);
    gpuMallocAsync(&result.histogram1, histogram_bins * sizeof(unsigned long long), stream);
    gpuMallocAsync(&result.histogram2, histogram_bins * sizeof(unsigned long long), stream);

    gpuMallocAsync(&result.min_sigma, sizeof(float), stream);
    gpuMallocAsync(&result.max_sigma, sizeof(float), stream);
    gpuMallocAsync(&result.median_noise, sizeof(float), stream);
    gpuMallocAsync(&result.rms_floor, sizeof(float), stream);
    gpuMallocAsync(&result.global_mad, sizeof(float), stream);
  }
  catch (...)
  {
    destroy_band_result(result, stream);
    throw;
  }

  return result;
}

void destroy_band_result(GpuBandResult &result, GpuStream stream)
{
  free_buffer(result.peak_value, stream);
  free_buffer(result.peak_local_noise, stream);
  free_buffer(result.peak_y, stream);
  free_buffer(result.peak_x, stream);
  free_buffer(result.island_size, stream);

  free_buffer(result.stamp_extreme, stream);
  free_buffer(result.donut_std, stream);
  free_buffer(result.donut_lmad, stream);
  free_buffer(result.beam_chi2, stream);
  free_buffer(result.flux, stream);
  free_buffer(result.boost, stream);
  free_buffer(result.boost_found, stream);

  free_buffer(result.noise_at_peak, stream);
  free_buffer(result.rms_at_peak, stream);
  free_buffer(result.selfnoise_at_peak, stream);

  free_buffer(result.histogram0, stream);
  free_buffer(result.histogram1, stream);
  free_buffer(result.histogram2, stream);

  free_buffer(result.min_sigma, stream);
  free_buffer(result.max_sigma, stream);
  free_buffer(result.median_noise, stream);
  free_buffer(result.rms_floor, stream);
  free_buffer(result.global_mad, stream);

  result = GpuBandResult{};
}
} // namespace breifast::gpu
