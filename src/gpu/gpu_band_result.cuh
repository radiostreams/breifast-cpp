#pragma once

#include <cstddef>
#include <vector>

#include "gpu_band_workspace.cuh"
#include "gpu_process_band.cuh"

namespace breifast::gpu
{
/// Host copy of a GpuBandResult: every island column holds label_count entries.
struct GpuBandHostResult
{
  std::size_t label_count = 0;

  IslandColumns<std::vector> islands;

  std::vector<unsigned long long> histogram0;
  std::vector<unsigned long long> histogram1;
  std::vector<unsigned long long> histogram2;

  float min_sigma = 0.F;
  float max_sigma = 0.F;
  float median_noise = 0.F;
  float rms_floor = 0.F;
  float global_mad = 0.F;
};

/// Copies the band outputs of `source` into `destination` and waits for `stream`.
void copy_band_result(const GpuBandResult &source, GpuBandHostResult &destination, GpuStream stream);
} // namespace breifast::gpu
