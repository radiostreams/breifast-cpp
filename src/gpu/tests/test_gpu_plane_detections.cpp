#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <stdexcept>
#include <optional>
#include <vector>

#include "gpu_plane_result.hpp"

using breifast::DetectionArrays;
using breifast::gpu::assemble_plane_detections;
using breifast::gpu::gpu_process_plane_result;
using breifast::gpu::GpuBandHostResult;
using breifast::gpu::GpuBandLabels;
using breifast::gpu::GpuBandStatus;
using breifast::gpu::GpuPlaneHostResult;
using Catch::Approx;
using Catch::Matchers::Equals;

namespace
{
GpuBandHostResult band_with(const std::size_t count, const float median_noise, const float rms_floor,
                            const float global_mad)
{
  GpuBandHostResult band;
  band.label_count = count;
  band.median_noise = median_noise;
  band.rms_floor = rms_floor;
  band.global_mad = global_mad;
  return band;
}

void resize_detection_vectors(GpuBandHostResult &band)
{
  const auto count = band.label_count;
  band.peak_value.resize(count);
  band.peak_local_noise.resize(count);
  band.peak_y.resize(count);
  band.peak_x.resize(count);
  band.island_size.resize(count);
  band.stamp_extreme.resize(count);
  band.donut_std.resize(count);
  band.donut_lmad.resize(count);
  band.beam_chi2.resize(count);
  band.flux.resize(count);
  band.boost.resize(count);
  band.noise_at_peak.resize(count);
  band.rms_at_peak.resize(count);
  band.selfnoise_at_peak.resize(count);
}
} // namespace

TEST_CASE("assemble_plane_detections appends bands in CPU reverse order", "[gpu][plane_detections]")
{
  GpuPlaneHostResult plane;
  plane.statuses = {GpuBandStatus::no_detections, GpuBandStatus::detections, GpuBandStatus::skipped};
  plane.bands.resize(plane.statuses.size());

  auto &low = plane.bands[0];
  low = band_with(1, 2.F, 0.5F, 4.F);
  resize_detection_vectors(low);
  low.peak_value[0] = 5.F;
  low.peak_local_noise[0] = 2.F;
  low.peak_y[0] = 1;
  low.peak_x[0] = 2;
  low.island_size[0] = 3;
  low.stamp_extreme[0] = 6.F;
  low.donut_std[0] = 2.5F;
  low.donut_lmad[0] = 1.5F;
  low.beam_chi2[0] = 0.25F;
  low.flux[0] = 10.F;
  low.boost[0] = 1.5F;
  low.noise_at_peak[0] = 3.F;
  low.rms_at_peak[0] = 2.F;
  low.selfnoise_at_peak[0] = 7.F;

  auto &high = plane.bands[1];
  high = band_with(2, 5.F, 1.F, 10.F);
  resize_detection_vectors(high);
  high.peak_value = {8.F, 1.F};
  high.peak_local_noise = {4.F, 1.F};
  high.peak_y = {3, 6};
  high.peak_x = {4, 7};
  high.island_size = {5, 8};
  high.stamp_extreme = {0.F, 2.F};
  high.donut_std = {0.F, 1.F};
  high.donut_lmad = {0.F, 1.F};
  high.beam_chi2 = {0.F, 0.5F};
  high.flux = {20.F, 2.F};
  high.boost = {0.F, 0.25F};
  high.noise_at_peak = {8.F, 1.F};
  high.rms_at_peak = {0.F, 4.F};
  high.selfnoise_at_peak = {9.F, 3.F};

  const std::vector<GpuBandLabels> labels = {{"I", "low"}, {"I", "high"}, {"I", "skipped"}};
  DetectionArrays detections;
  assemble_plane_detections(plane, labels, 2.F, true, detections);

  REQUIRE(detections.size() == 3);
  CHECK_THAT(detections.band_label(), Equals(std::vector<std::string>{"high", "high", "low"}));
  CHECK_THAT(detections.peak_y(), Equals(std::vector<long>{3, 6, 1}));
  CHECK_THAT(detections.peak_x(), Equals(std::vector<long>{4, 7, 2}));
  CHECK_THAT(detections.island_size(), Equals(std::vector<long>{5, 8, 3}));

  CHECK(detections.flux()[0] == Approx(20.F));
  CHECK(detections.local_noise_median_ratio()[0] == Approx(4.F / 5.F));
  CHECK(detections.signal_to_noise()[0] == Approx(8.F));
  CHECK(detections.signal_to_noise_rms()[0] == Approx(0.F));
  CHECK(detections.signal_to_noise_extended_box()[0] == Approx(0.F));
  CHECK(detections.signal_to_noise_local_mad()[0] == Approx(0.F));
  CHECK(detections.signal_to_noise_global_mad()[0] == Approx(2.F));
  CHECK(detections.self_noise()[0] == Approx(9.F));
  CHECK(detections.excess_noise_factor()[0] == Approx(8.F));

  CHECK(detections.flux()[1] == Approx(2.F));
  CHECK(detections.local_noise_median_ratio()[1] == Approx(1.F / 5.F));
  CHECK(detections.signal_to_noise_rms()[1] == Approx(2.F));
  CHECK(detections.signal_to_noise_extended_box()[1] == Approx(2.F));
  CHECK(detections.signal_to_noise_local_mad()[1] == Approx(2.F));
  CHECK(detections.signal_to_noise_global_mad()[1] == Approx(0.2F));
  CHECK(detections.excess_noise_factor()[1] == Approx(0.25F));
  CHECK(detections.beam_chi2()[1] == Approx(0.5F));
  CHECK(detections.boost()[1] == Approx(0.25F));

  CHECK(detections.local_noise_median_ratio()[2] == Approx(1.F));
  CHECK(detections.signal_to_noise()[2] == Approx(5.F));
  CHECK(detections.signal_to_noise_rms()[2] == Approx(4.F));
  CHECK(detections.signal_to_noise_extended_box()[2] == Approx(10.F / 3.F));
  CHECK(detections.signal_to_noise_local_mad()[2] == Approx(10.F / 1.5F));
  CHECK(detections.signal_to_noise_global_mad()[2] == Approx(2.5F));
  CHECK(detections.self_noise()[2] == Approx(7.F));
  CHECK(detections.excess_noise_factor()[2] == Approx(1.5F));
  CHECK(detections.beam_chi2()[2] == Approx(0.25F));
  CHECK(detections.boost()[2] == Approx(1.5F));
}

TEST_CASE("assemble_plane_detections ignores skipped bands", "[gpu][plane_detections]")
{
  GpuPlaneHostResult plane;
  plane.statuses = {GpuBandStatus::skipped, GpuBandStatus::skipped};
  plane.bands.resize(plane.statuses.size());

  const std::vector<GpuBandLabels> labels = {{"I", "first"}, {"I", "second"}};
  DetectionArrays detections;
  assemble_plane_detections(plane, labels, 2.F, true, detections);

  CHECK(detections.size() == 0);
}

TEST_CASE("assemble_plane_detections validates its inputs", "[gpu][plane_detections]")
{
  GpuPlaneHostResult plane;
  plane.statuses = {GpuBandStatus::skipped, GpuBandStatus::skipped};
  plane.bands.resize(plane.statuses.size());

  const std::vector<GpuBandLabels> one_label = {{"I", "first"}};
  DetectionArrays detections;
  CHECK_THROWS_AS(assemble_plane_detections(plane, one_label, 2.F, true, detections), std::runtime_error);

  const std::vector<GpuBandLabels> labels = {{"I", "first"}, {"I", "second"}};
  CHECK_THROWS_AS(assemble_plane_detections(plane, labels, -1.F, true, detections), std::runtime_error);
}

TEST_CASE("assemble_plane_result builds the final ProcessPlaneResult", "[gpu][plane_detections]")
{
  GpuPlaneHostResult plane;
  plane.statuses = {GpuBandStatus::no_detections};
  plane.bands.resize(1);
  plane.min_sigma = 0.5F;
  plane.max_sigma = 2.5F;
  plane.histograms[0] = {1ULL, 2ULL};
  plane.histograms[1] = {3ULL, 4ULL};
  plane.histograms[2] = {5ULL, 6ULL};
  plane.median_noises = {2.0};

  auto &band = plane.bands[0];
  band = band_with(1, 2.F, 0.5F, 4.F);
  resize_detection_vectors(band);
  band.peak_value[0] = 5.F;
  band.peak_local_noise[0] = 2.F;
  band.peak_y[0] = 1;
  band.peak_x[0] = 2;
  band.island_size[0] = 3;
  band.stamp_extreme[0] = 6.F;
  band.donut_std[0] = 2.5F;
  band.donut_lmad[0] = 1.5F;
  band.flux[0] = 10.F;
  band.noise_at_peak[0] = 3.F;
  band.rms_at_peak[0] = 2.F;
  band.selfnoise_at_peak[0] = 7.F;

  const std::vector<GpuBandLabels> labels = {{"I", "band"}};
  breifast::SpectralVarianceStats variance;
  const auto result =
      assemble_plane_result(plane, labels, 2.F, true, std::optional<breifast::SpectralVarianceStats>{variance});

  CHECK(result.any_data == false);
  CHECK(result.detections.size() == 1);
  CHECK_THAT(result.detections.band_label(), Equals(std::vector<std::string>{"band"}));
  REQUIRE(result.min_sigma.has_value() == true);
  REQUIRE(result.max_sigma.has_value() == true);
  CHECK(*result.min_sigma == 0.5F);
  CHECK(*result.max_sigma == 2.5F);
  CHECK_THAT(result.histograms[0], Equals(std::vector<long>{1, 2}));
  CHECK_THAT(result.histograms[1], Equals(std::vector<long>{3, 4}));
  CHECK_THAT(result.histograms[2], Equals(std::vector<long>{5, 6}));
  CHECK_THAT(result.median_noises, Equals(std::vector<double>{2.0}));
  CHECK(result.variance_stats.has_value() == true);

  const auto without_variance = assemble_plane_result(plane, labels, 2.F, true, std::nullopt);
  CHECK(without_variance.variance_stats.has_value() == false);
}

TEST_CASE("gpu_process_plane_result wraps jobs into a ProcessPlaneResult", "[gpu][plane_detections]")
{
  std::vector<breifast::gpu::GpuBandJob> jobs;
  const std::vector<GpuBandLabels> labels;

  const auto result = gpu_process_plane_result(jobs, labels, 2.F, true, std::nullopt);
  CHECK(result.detections.size() == 0);
  CHECK(result.histograms[0].empty());
  CHECK(result.histograms[1].empty());
  CHECK(result.histograms[2].empty());
  CHECK(result.median_noises.empty());
  CHECK(result.min_sigma.has_value() == false);
  CHECK(result.max_sigma.has_value() == false);
  CHECK(result.any_data == false);
  CHECK(result.variance_stats.has_value() == false);

  breifast::SpectralVarianceStats variance;
  const auto with_variance =
      gpu_process_plane_result(jobs, labels, 2.F, true, std::optional<breifast::SpectralVarianceStats>{variance});
  CHECK(with_variance.variance_stats.has_value() == true);
}
