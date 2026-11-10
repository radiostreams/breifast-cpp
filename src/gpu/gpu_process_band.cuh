#pragma once

#include <cstddef>
#include <exception>
#include <vector>

#include "gpu_band_workspace.cuh"

namespace breifast::gpu
{
enum class GpuBandStatus
{
  skipped,
  no_detections,
  detections
};

GpuBandStatus gpu_process_band(const GpuBandInput &input, GpuBandWorkspace &workspace, GpuBandResult &result,
                               GpuStream stream);

struct GpuBandJob
{
  GpuBandInput input;
  GpuBandWorkspaceSizes workspace_sizes;
  std::size_t result_n_pixels = 0;
  std::size_t result_histogram_bins = 0;

  GpuBandWorkspace workspace;
  GpuBandResult result;
  GpuStream stream{};
  GpuBandStatus status = GpuBandStatus::skipped;
  std::exception_ptr error;
};

void gpu_process_bands(std::vector<GpuBandJob> &jobs);
void destroy_band_job(GpuBandJob &job) noexcept;
} // namespace breifast::gpu
