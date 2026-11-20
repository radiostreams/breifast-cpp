/**
 * @file process_plane_result.hpp
 * @brief Per-plane output structs shared by the CPU and GPU plane-processing paths.
 */
#pragma once

#include <array>
#include <optional>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "detection_arrays.hpp"

namespace breifast
{
/**
 * @brief Per-pixel spectral-variance QA diagnostics accumulated across a plane's frequency channels (not used for
 *        detection, only diagnostic aggregation further downstream in breifast.py).
 */
struct SpectralVarianceStats
{
  xt::xtensor<float, 2> sum_pow2;
  xt::xtensor<float, 2> sum_pow3;
  xt::xtensor<float, 2> sum_pow4;
  xt::xtensor<bool, 1> nonzero_channels_mask;
  std::optional<xt::xtensor<float, 2>> per_channels_squared;
};

/**
 * @brief Everything process_plane() produces for one plane: detections, sigma range, per-band histograms, and (if
 *        multi-channel) variance QA stats.
 */
struct ProcessPlaneResult
{
  bool any_data = false;
  DetectionArrays detections;
  std::optional<float> min_sigma;
  std::optional<float> max_sigma;
  std::array<std::vector<long>, 3> histograms;
  std::vector<double> median_noises;
  std::optional<SpectralVarianceStats> variance_stats;
};
} // namespace breifast
