#include "gpu_band_workspace.cuh"

#include <stdexcept>
#include <string>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
namespace
{
template <typename T> void free_buffer(T *&buffer, GpuStream stream) noexcept
{
  if (buffer != nullptr)
  {
    // Cleanup must never throw: freeing runs after a plane may have failed, and a
    // sticky CUDA error would otherwise abort the remaining frees and leak them.
    (void)cudaFree(buffer);
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
    gpuMalloc(&workspace.plane, n_pixels * sizeof(float));
    gpuMalloc(&workspace.noise_map, n_pixels * sizeof(float));
    gpuMalloc(&workspace.snr, n_pixels * sizeof(float));
    gpuMalloc(&workspace.mask, n_pixels * sizeof(bool));
    gpuMalloc(&workspace.row_sums, n_pixels * sizeof(float));
    gpuMalloc(&workspace.rms_map, n_pixels * sizeof(float));

    gpuMalloc(&workspace.labels, n_pixels * sizeof(unsigned int));
    gpuMalloc(&workspace.label_scratch, n_pixels * sizeof(unsigned int));

    gpuMalloc(&workspace.donut_stamps, sizes.stamp_capacity * sizes.stamp_pixels * sizeof(float));
    if (sizes.beam_stamp_pixels > 0)
    {
      gpuMalloc(&workspace.beam_stamps, sizes.stamp_capacity * sizes.beam_stamp_pixels * sizeof(float));
    }

    gpuMalloc(&workspace.any_flag, sizeof(int));
    gpuMalloc(&workspace.label_count, sizeof(unsigned int));
  }
  catch (...)
  {
    destroy_band_workspace(workspace, stream);
    throw;
  }

  return workspace;
}

void destroy_band_workspace(GpuBandWorkspace &workspace, GpuStream stream) noexcept
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
    gpuMalloc(&result.peak_value, n_pixels * sizeof(float));
    gpuMalloc(&result.peak_local_noise, n_pixels * sizeof(float));
    gpuMalloc(&result.peak_y, n_pixels * sizeof(long));
    gpuMalloc(&result.peak_x, n_pixels * sizeof(long));
    gpuMalloc(&result.island_size, n_pixels * sizeof(unsigned int));

    gpuMalloc(&result.stamp_extreme, n_pixels * sizeof(float));
    gpuMalloc(&result.donut_std, n_pixels * sizeof(float));
    gpuMalloc(&result.donut_lmad, n_pixels * sizeof(float));
    gpuMalloc(&result.beam_chi2, n_pixels * sizeof(float));
    gpuMalloc(&result.flux, n_pixels * sizeof(float));
    gpuMalloc(&result.boost, n_pixels * sizeof(float));
    gpuMalloc(&result.boost_found, n_pixels * sizeof(bool));

    gpuMalloc(&result.noise_at_peak, n_pixels * sizeof(float));
    gpuMalloc(&result.rms_at_peak, n_pixels * sizeof(float));
    gpuMalloc(&result.selfnoise_at_peak, n_pixels * sizeof(float));

    gpuMalloc(&result.histogram0, histogram_bins * sizeof(unsigned long long));
    gpuMalloc(&result.histogram1, histogram_bins * sizeof(unsigned long long));
    gpuMalloc(&result.histogram2, histogram_bins * sizeof(unsigned long long));

    gpuMalloc(&result.min_sigma, sizeof(float));
    gpuMalloc(&result.max_sigma, sizeof(float));
    gpuMalloc(&result.median_noise, sizeof(float));
    gpuMalloc(&result.rms_floor, sizeof(float));
    gpuMalloc(&result.global_mad, sizeof(float));
  }
  catch (...)
  {
    destroy_band_result(result, stream);
    throw;
  }

  return result;
}

void destroy_band_result(GpuBandResult &result, GpuStream stream) noexcept
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
