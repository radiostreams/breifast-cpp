#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "gpu_band_result.cuh"
#include "gpu_test_utils.hpp"

using breifast::gpu::copy_band_result;
using breifast::gpu::GpuBandHostResult;
using breifast::gpu::GpuBandResult;
using breifast::gpu::Stream;
using breifast::gpu::test::upload;
using Catch::Matchers::Equals;

TEST_CASE("copy_band_result copies per-island arrays, histograms and scalars", "[gpu][band_result]")
{
  constexpr std::size_t histogram_bins = 3;
  constexpr unsigned int label_count = 2;

  GpuBandResult source;
  source.ensure(label_count, histogram_bins);
  source.label_count = label_count;

  const std::vector<float> floats = {1.F, 2.F};
  const std::vector<long> coordinates = {3L, 4L};
  const std::vector<unsigned int> sizes = {5U, 6U};
  const std::vector<unsigned long long> histogram0 = {10ULL, 11ULL, 12ULL};
  const std::vector<unsigned long long> histogram1 = {20ULL, 21ULL, 22ULL};
  const std::vector<unsigned long long> histogram2 = {30ULL, 31ULL, 32ULL};

  source.islands.peak_value = upload(floats);
  source.islands.peak_local_noise = upload(floats);
  source.islands.peak_y = upload(coordinates);
  source.islands.peak_x = upload(coordinates);
  source.islands.island_size = upload(sizes);
  source.islands.stamp_extreme = upload(floats);
  source.islands.donut_std = upload(floats);
  source.islands.donut_lmad = upload(floats);
  source.islands.beam_chi2 = upload(floats);
  source.islands.flux = upload(floats);
  source.islands.boost = upload(floats);
  source.islands.noise_at_peak = upload(floats);
  source.islands.rms_at_peak = upload(floats);
  source.islands.selfnoise_at_peak = upload(floats);
  source.histogram0 = upload(histogram0);
  source.histogram1 = upload(histogram1);
  source.histogram2 = upload(histogram2);

  const float min_sigma = 0.1F;
  const float max_sigma = 0.2F;
  const float median_noise = 0.3F;
  const float rms_floor = 0.4F;
  const float global_mad = 0.5F;
  source.min_sigma = upload(std::vector<float>{min_sigma});
  source.max_sigma = upload(std::vector<float>{max_sigma});
  source.median_noise = upload(std::vector<float>{median_noise});
  source.rms_floor = upload(std::vector<float>{rms_floor});
  source.global_mad = upload(std::vector<float>{global_mad});

  GpuBandHostResult destination;
  const Stream stream;
  copy_band_result(source, destination, stream.get());

  CHECK(destination.label_count == label_count);
  CHECK_THAT(destination.islands.peak_value, Equals(floats));
  CHECK_THAT(destination.islands.peak_local_noise, Equals(floats));
  CHECK_THAT(destination.islands.peak_y, Equals(coordinates));
  CHECK_THAT(destination.islands.peak_x, Equals(coordinates));
  CHECK_THAT(destination.islands.island_size, Equals(sizes));
  CHECK_THAT(destination.islands.stamp_extreme, Equals(floats));
  CHECK_THAT(destination.islands.donut_std, Equals(floats));
  CHECK_THAT(destination.islands.donut_lmad, Equals(floats));
  CHECK_THAT(destination.islands.beam_chi2, Equals(floats));
  CHECK_THAT(destination.islands.flux, Equals(floats));
  CHECK_THAT(destination.islands.boost, Equals(floats));
  CHECK_THAT(destination.islands.noise_at_peak, Equals(floats));
  CHECK_THAT(destination.islands.rms_at_peak, Equals(floats));
  CHECK_THAT(destination.islands.selfnoise_at_peak, Equals(floats));
  CHECK_THAT(destination.histogram0, Equals(histogram0));
  CHECK_THAT(destination.histogram1, Equals(histogram1));
  CHECK_THAT(destination.histogram2, Equals(histogram2));
  CHECK(destination.min_sigma == min_sigma);
  CHECK(destination.max_sigma == max_sigma);
  CHECK(destination.median_noise == median_noise);
  CHECK(destination.rms_floor == rms_floor);
  CHECK(destination.global_mad == global_mad);
}
