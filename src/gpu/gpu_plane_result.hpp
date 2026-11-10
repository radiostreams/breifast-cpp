#pragma once

#include <optional>
#include <vector>

#include "gpu_plane_detections.hpp"
#include "process_plane_result.hpp"

namespace breifast::gpu
{
breifast::ProcessPlaneResult assemble_plane_result(const GpuPlaneHostResult &plane,
                                                   const std::vector<GpuBandLabels> &labels, float max_to_std_ratio,
                                                   bool use_rms_noise_estimator,
                                                   std::optional<breifast::SpectralVarianceStats> variance_stats);

breifast::ProcessPlaneResult gpu_process_plane_result(
    std::vector<GpuBandJob> &jobs, const std::vector<GpuBandLabels> &labels, float max_to_std_ratio,
    bool use_rms_noise_estimator, std::optional<breifast::SpectralVarianceStats> variance_stats);
} // namespace breifast::gpu
