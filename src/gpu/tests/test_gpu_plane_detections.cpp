#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <limits>
#include <string>
#include <vector>

#include "gpu_plane_detections.hpp"
#include "process_plane_result.hpp"

using breifast::ProcessPlaneResult;
using breifast::gpu::append_band_result;
using breifast::gpu::GpuBandHostResult;
using breifast::gpu::GpuBandLabels;
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

TEST_CASE("append_band_result appends detections in the order the bands are appended", "[gpu][plane_detections]")
{
  GpuBandHostResult low = band_with(1, 2.F, 0.5F, 4.F);
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

  GpuBandHostResult high = band_with(2, 5.F, 1.F, 10.F);
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

  // CPU order: the last band first.
  ProcessPlaneResult result;
  append_band_result(high, {"I", "high"}, 2.F, true, result);
  append_band_result(low, {"I", "low"}, 2.F, true, result);
  const auto &detections = result.detections;

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

TEST_CASE("append_band_result keeps the CPU histogram, median and sigma order", "[gpu][plane_detections]")
{
  // The last band, appended first, passed the filters with sigmas in [0.5, 2]; the first band, appended last, had no
  // pixel pass (infinite minimum) and leaves the range alone, but its histograms are the ones the plane keeps.
  GpuBandHostResult last_band;
  last_band.histogram0 = {7ULL, 8ULL};
  last_band.histogram1 = {9ULL, 10ULL};
  last_band.histogram2 = {11ULL, 12ULL};
  last_band.median_noise = 20.F;
  last_band.min_sigma = 0.5F;
  last_band.max_sigma = 2.F;

  GpuBandHostResult first_band;
  first_band.histogram0 = {1ULL, 2ULL};
  first_band.histogram1 = {3ULL, 4ULL};
  first_band.histogram2 = {5ULL, 6ULL};
  first_band.median_noise = 10.F;
  first_band.min_sigma = std::numeric_limits<float>::infinity();
  first_band.max_sigma = std::numeric_limits<float>::lowest();

  ProcessPlaneResult result;
  append_band_result(last_band, {"I", "last"}, 2.F, true, result);
  append_band_result(first_band, {"I", "first"}, 2.F, true, result);

  CHECK(result.detections.size() == 0);
  CHECK_THAT(result.histograms[0], Equals(std::vector<long>{1, 2}));
  CHECK_THAT(result.histograms[1], Equals(std::vector<long>{3, 4}));
  CHECK_THAT(result.histograms[2], Equals(std::vector<long>{5, 6}));
  CHECK_THAT(result.median_noises, Equals(std::vector<double>{20.0, 10.0}));
  REQUIRE(result.min_sigma.has_value() == true);
  REQUIRE(result.max_sigma.has_value() == true);
  CHECK(*result.min_sigma == 0.5F);
  CHECK(*result.max_sigma == 2.F);
}
