#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "gpu_buffers.hpp"
#include "gpu_pipeline.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::GpuSpectralVarianceHostResult;
using breifast::gpu::gpu_spectral_variance_stats_host;
using Catch::Approx;
using Catch::Matchers::Equals;

TEST_CASE("gpu_spectral_variance_stats_host copies moments and mask", "[gpu][spectral_variance]")
{
  constexpr std::size_t n_y = 2;
  constexpr std::size_t n_x = 2;
  constexpr std::size_t n_channels = 3;
  constexpr std::size_t n_pixels = n_y * n_x;

  const std::vector<float> cube_host = {
      1.F, 2.F, 3.F, 4.F,    // channel 0, selected
      10.F, 20.F, 30.F, 40.F, // channel 1, skipped
      2.F, 3.F, 4.F, 5.F,    // channel 2, selected
  };
  const std::vector<std::uint8_t> mask_host = {1, 0, 1};

  DeviceVector<float> cube(cube_host.begin(), cube_host.end());
  DeviceVector<std::uint8_t> mask(mask_host.begin(), mask_host.end());

  GpuSpectralVarianceHostResult result;
  gpu_spectral_variance_stats_host(thrust::raw_pointer_cast(cube.data()),
                                   reinterpret_cast<const bool *>(thrust::raw_pointer_cast(mask.data())), n_y, n_x,
                                   n_channels, result);

  REQUIRE(result.sum_pow2.size() == n_pixels);
  REQUIRE(result.sum_pow3.size() == n_pixels);
  REQUIRE(result.sum_pow4.size() == n_pixels);
  REQUIRE(result.nonzero_channels_mask.size() == n_channels);

  const std::vector<float> expected_sum2 = {5.F, 13.F, 25.F, 41.F};
  const std::vector<float> expected_sum3 = {9.F, 35.F, 91.F, 189.F};
  const std::vector<float> expected_sum4 = {17.F, 97.F, 337.F, 881.F};
  for (std::size_t i = 0; i < n_pixels; ++i)
  {
    CHECK(result.sum_pow2[i] == Approx(expected_sum2[i]));
    CHECK(result.sum_pow3[i] == Approx(expected_sum3[i]));
    CHECK(result.sum_pow4[i] == Approx(expected_sum4[i]));
  }
  CHECK_THAT(result.nonzero_channels_mask, Equals(std::vector<unsigned char>{1, 0, 1}));
}

TEST_CASE("gpu_spectral_variance_stats_host validates dimensions", "[gpu][spectral_variance]")
{
  GpuSpectralVarianceHostResult result;
  CHECK_THROWS_AS(gpu_spectral_variance_stats_host(nullptr, nullptr, 0, 1, 1, result), std::runtime_error);
  CHECK_THROWS_AS(gpu_spectral_variance_stats_host(nullptr, nullptr, 1, 1, 1, result), std::runtime_error);
}
