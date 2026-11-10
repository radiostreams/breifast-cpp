#pragma once

#include <cstddef>
#include <exception>
#include <optional>
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
  GpuBandJob(GpuBandInput input, const GpuBandWorkspaceSizes &workspace_sizes, std::size_t result_n_pixels,
             std::size_t result_histogram_bins);

  GpuBandJob(const GpuBandJob &) = delete;
  GpuBandJob &operator=(const GpuBandJob &) = delete;
  GpuBandJob(GpuBandJob &&) noexcept;
  GpuBandJob &operator=(GpuBandJob &&) noexcept;
  ~GpuBandJob();

  GpuBandInput input;
  GpuBandWorkspaceSizes workspace_sizes;
  std::size_t result_n_pixels = 0;
  std::size_t result_histogram_bins = 0;
  std::optional<GpuBandWorkspace> workspace;
  std::optional<GpuBandResult> result;
  GpuStream stream{};
  GpuBandStatus status = GpuBandStatus::skipped;
  std::exception_ptr error;
};

void gpu_process_bands(std::vector<GpuBandJob> &jobs);
} // namespace breifast::gpu
