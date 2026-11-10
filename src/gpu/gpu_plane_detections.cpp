#include "gpu_plane_result.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace breifast::gpu
{
void assemble_plane_detections(const GpuPlaneHostResult &plane, const std::vector<GpuBandLabels> &labels,
                               float max_to_std_ratio, bool use_rms_noise_estimator,
                               breifast::DetectionArrays &detections)
{
  if (labels.size() != plane.statuses.size() || labels.size() != plane.bands.size())
  {
    throw std::runtime_error("assemble_plane_detections: labels, statuses and bands size mismatch");
  }
  if (max_to_std_ratio < 0.F)
  {
    throw std::runtime_error("assemble_plane_detections: max_to_std_ratio must be non-negative");
  }

  for (size_t i = plane.statuses.size(); i > 0; --i)
  {
    const size_t band_index = i - 1;
    if (plane.statuses[band_index] == GpuBandStatus::skipped)
    {
      continue;
    }

    const auto &band = plane.bands[band_index];
    const size_t count = band.label_count;
    const auto require_size = [count](const auto &values, const char *name)
    {
      if (values.size() < count)
      {
        throw std::runtime_error(std::string("assemble_plane_detections: truncated ") + name);
      }
    };
    require_size(band.peak_value, "peak_value");
    require_size(band.peak_local_noise, "peak_local_noise");
    require_size(band.peak_y, "peak_y");
    require_size(band.peak_x, "peak_x");
    require_size(band.island_size, "island_size");
    require_size(band.stamp_extreme, "stamp_extreme");
    require_size(band.donut_std, "donut_std");
    require_size(band.donut_lmad, "donut_lmad");
    require_size(band.beam_chi2, "beam_chi2");
    require_size(band.flux, "flux");
    require_size(band.boost, "boost");
    require_size(band.noise_at_peak, "noise_at_peak");
    require_size(band.rms_at_peak, "rms_at_peak");
    require_size(band.selfnoise_at_peak, "selfnoise_at_peak");

    detections.reserve(detections.size() + count);
    for (size_t j = 0; j < count; ++j)
    {
      const float flux = band.flux[j];
      const float rms = band.donut_std[j];
      const float lmad = band.donut_lmad[j];
      const float adjacent_noise = band.stamp_extreme[j] / max_to_std_ratio;
      const float noisemap0_at_peak =
          use_rms_noise_estimator ? std::max(band.rms_at_peak[j], band.rms_floor) : band.global_mad;

      detections.add(band.peak_y[j], band.peak_x[j], static_cast<long>(band.island_size[j]),
                     labels[band_index].stokes_label, labels[band_index].band_label, flux,
                     band.peak_local_noise[j], band.peak_local_noise[j] / band.median_noise, band.peak_value[j],
                     rms != 0.F ? flux / rms : 0.F, adjacent_noise != 0.F ? flux / adjacent_noise : 0.F,
                     lmad != 0.F ? flux / lmad : 0.F, flux / band.global_mad, band.selfnoise_at_peak[j],
                     band.noise_at_peak[j] / noisemap0_at_peak, band.beam_chi2[j], band.boost[j]);
    }
  }
}

breifast::ProcessPlaneResult assemble_plane_result(const GpuPlaneHostResult &plane,
                                                   const std::vector<GpuBandLabels> &labels, float max_to_std_ratio,
                                                   bool use_rms_noise_estimator,
                                                   std::optional<breifast::SpectralVarianceStats> variance_stats)
{
  breifast::ProcessPlaneResult result;
  assemble_plane_detections(plane, labels, max_to_std_ratio, use_rms_noise_estimator, result.detections);

  result.min_sigma = plane.min_sigma;
  result.max_sigma = plane.max_sigma;
  for (size_t i = 0; i < result.histograms.size(); ++i)
  {
    result.histograms[i].resize(plane.histograms[i].size());
    std::transform(plane.histograms[i].begin(), plane.histograms[i].end(), result.histograms[i].begin(),
                   [](const unsigned long long value) { return static_cast<long>(value); });
  }
  result.median_noises = plane.median_noises;
  result.variance_stats = std::move(variance_stats);
  return result;
}

breifast::ProcessPlaneResult gpu_process_plane_result(
    std::vector<GpuBandJob> &jobs, const std::vector<GpuBandLabels> &labels, float max_to_std_ratio,
    bool use_rms_noise_estimator, std::optional<breifast::SpectralVarianceStats> variance_stats)
{
  GpuPlaneHostResult plane;
  gpu_process_plane(jobs, plane);
  return assemble_plane_result(plane, labels, max_to_std_ratio, use_rms_noise_estimator, std::move(variance_stats));
}
} // namespace breifast::gpu
