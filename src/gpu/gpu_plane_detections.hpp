#pragma once

#include <string>
#include <vector>

#include "detection_arrays.hpp"
#include "gpu_band_result.cuh"

namespace breifast::gpu
{
struct GpuBandLabels
{
  std::string stokes_label;
  std::string band_label;
};

void assemble_plane_detections(const GpuPlaneHostResult &plane, const std::vector<GpuBandLabels> &labels,
                               float max_to_std_ratio, bool use_rms_noise_estimator,
                               breifast::DetectionArrays &detections);

} // namespace breifast::gpu
