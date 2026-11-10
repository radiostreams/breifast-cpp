#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "gpu_band_result.cuh"
#include "device_vector.hpp"
#include "gpu_runtime.hpp"

using breifast::gpu::assemble_plane_bands;
using breifast::gpu::copy_band_result;
using breifast::gpu::GpuBandHostResult;
using breifast::gpu::GpuBandStatus;
using breifast::gpu::GpuPlaneHostResult;
using breifast::gpu::GpuBandResult;
using Catch::Matchers::Equals;

TEST_CASE("copy_band_result copies per-island arrays, histograms and scalars", "[gpu][band_result]")
{
  constexpr std::size_t n_pixels = 4;
  constexpr std::size_t histogram_bins = 3;
  constexpr unsigned int label_count = 2;

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  GpuBandResult source(n_pixels, histogram_bins);
  source.label_count = label_count;

  const std::vector<float> floats = {1.F, 2.F};
  const std::vector<long> coordinates = {3L, 4L};
  const std::vector<unsigned int> sizes = {5U, 6U};
  const std::vector<unsigned char> found = {1U, 0U};
  const std::vector<unsigned long long> histogram0 = {10ULL, 11ULL, 12ULL};
  const std::vector<unsigned long long> histogram1 = {20ULL, 21ULL, 22ULL};
  const std::vector<unsigned long long> histogram2 = {30ULL, 31ULL, 32ULL};

  gpuMemcpyHostToDeviceAsync(source.peak_value, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.peak_local_noise, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.peak_y, coordinates.data(), label_count * sizeof(long), stream);
  gpuMemcpyHostToDeviceAsync(source.peak_x, coordinates.data(), label_count * sizeof(long), stream);
  gpuMemcpyHostToDeviceAsync(source.island_size, sizes.data(), label_count * sizeof(unsigned int), stream);
  gpuMemcpyHostToDeviceAsync(source.stamp_extreme, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.donut_std, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.donut_lmad, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.beam_chi2, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.flux, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.boost, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.boost_found, found.data(), label_count * sizeof(bool), stream);
  gpuMemcpyHostToDeviceAsync(source.noise_at_peak, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.rms_at_peak, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.selfnoise_at_peak, floats.data(), label_count * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.histogram0, histogram0.data(), histogram_bins * sizeof(unsigned long long), stream);
  gpuMemcpyHostToDeviceAsync(source.histogram1, histogram1.data(), histogram_bins * sizeof(unsigned long long), stream);
  gpuMemcpyHostToDeviceAsync(source.histogram2, histogram2.data(), histogram_bins * sizeof(unsigned long long), stream);

  const float min_sigma = 0.1F;
  const float max_sigma = 0.2F;
  const float median_noise = 0.3F;
  const float rms_floor = 0.4F;
  const float global_mad = 0.5F;
  gpuMemcpyHostToDeviceAsync(source.min_sigma, &min_sigma, sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.max_sigma, &max_sigma, sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.median_noise, &median_noise, sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.rms_floor, &rms_floor, sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(source.global_mad, &global_mad, sizeof(float), stream);

  GpuBandHostResult destination;
  copy_band_result(source, destination, stream);

  CHECK(destination.label_count == label_count);
  CHECK_THAT(destination.peak_value, Equals(floats));
  CHECK_THAT(destination.peak_local_noise, Equals(floats));
  CHECK_THAT(destination.peak_y, Equals(coordinates));
  CHECK_THAT(destination.peak_x, Equals(coordinates));
  CHECK_THAT(destination.island_size, Equals(sizes));
  CHECK_THAT(destination.stamp_extreme, Equals(floats));
  CHECK_THAT(destination.donut_std, Equals(floats));
  CHECK_THAT(destination.donut_lmad, Equals(floats));
  CHECK_THAT(destination.beam_chi2, Equals(floats));
  CHECK_THAT(destination.flux, Equals(floats));
  CHECK_THAT(destination.boost, Equals(floats));
  CHECK_THAT(destination.noise_at_peak, Equals(floats));
  CHECK_THAT(destination.rms_at_peak, Equals(floats));
  CHECK_THAT(destination.selfnoise_at_peak, Equals(floats));
  CHECK(destination.boost_found == found);
  CHECK_THAT(destination.histogram0, Equals(histogram0));
  CHECK_THAT(destination.histogram1, Equals(histogram1));
  CHECK_THAT(destination.histogram2, Equals(histogram2));
  CHECK(destination.min_sigma == min_sigma);
  CHECK(destination.max_sigma == max_sigma);
  CHECK(destination.median_noise == median_noise);
  CHECK(destination.rms_floor == rms_floor);
  CHECK(destination.global_mad == global_mad);

  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);
}

TEST_CASE("assemble_plane_bands matches CPU histogram, median and sigma ordering", "[gpu][band_result]")
{
  const std::vector<GpuBandStatus> statuses = {
      GpuBandStatus::skipped,
      GpuBandStatus::no_detections,
      GpuBandStatus::detections,
  };

  std::vector<GpuBandHostResult> bands(3);
  bands[1].histogram0 = {1ULL, 2ULL};
  bands[1].histogram1 = {3ULL, 4ULL};
  bands[1].histogram2 = {5ULL, 6ULL};
  bands[1].median_noise = 10.0;
  bands[1].min_sigma = std::numeric_limits<float>::infinity();
  bands[1].max_sigma = std::numeric_limits<float>::lowest();

  bands[2].histogram0 = {7ULL, 8ULL};
  bands[2].histogram1 = {9ULL, 10ULL};
  bands[2].histogram2 = {11ULL, 12ULL};
  bands[2].median_noise = 20.0;
  bands[2].min_sigma = 0.5F;
  bands[2].max_sigma = 2.0F;

  GpuPlaneHostResult result;
  assemble_plane_bands(statuses, bands, result);

  CHECK(result.statuses == statuses);
  CHECK_THAT(result.histograms[0], Equals(std::vector<unsigned long long>{1ULL, 2ULL}));
  CHECK_THAT(result.histograms[1], Equals(std::vector<unsigned long long>{3ULL, 4ULL}));
  CHECK_THAT(result.histograms[2], Equals(std::vector<unsigned long long>{5ULL, 6ULL}));
  CHECK_THAT(result.median_noises, Equals(std::vector<double>{20.0, 10.0}));
  REQUIRE(result.min_sigma.has_value() == true);
  REQUIRE(result.max_sigma.has_value() == true);
  CHECK(*result.min_sigma == 0.5F);
  CHECK(*result.max_sigma == 2.0F);
}

TEST_CASE("assemble_plane_bands leaves everything empty when all bands are skipped", "[gpu][band_result]")
{
  const std::vector<GpuBandStatus> statuses = {
      GpuBandStatus::skipped,
      GpuBandStatus::skipped,
  };
  const std::vector<GpuBandHostResult> bands(2);

  GpuPlaneHostResult result;
  assemble_plane_bands(statuses, bands, result);

  CHECK(result.histograms[0].empty());
  CHECK(result.histograms[1].empty());
  CHECK(result.histograms[2].empty());
  CHECK(result.median_noises.empty());
  CHECK(result.min_sigma.has_value() == false);
  CHECK(result.max_sigma.has_value() == false);
}
