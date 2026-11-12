#pragma once

#include <string>

#include "gpu_band_result.cuh"

namespace breifast
{
struct ProcessPlaneResult; // process_plane_result.hpp includes xtensor, which nvcc cannot compile
}

namespace breifast::gpu
{
struct GpuBandLabels
{
  std::string stokes_label;
  std::string band_label;
};

/// Adds one processed band to `result` as the CPU process_band does: its histograms replace the previous band's, its
/// median noise is appended, its sigma range merged, and its islands appended as detections. Called in CPU band order.
void append_band_result(const GpuBandHostResult &band, const GpuBandLabels &labels, float max_to_std_ratio,
                        bool use_rms_noise_estimator, breifast::ProcessPlaneResult &result);
} // namespace breifast::gpu
