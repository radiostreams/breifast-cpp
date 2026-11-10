#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "gpu_spectral_variance.hpp"

using breifast::gpu::assemble_spectral_variance_stats;
using breifast::gpu::GpuSpectralVarianceHostResult;

TEST_CASE("assemble_spectral_variance_stats copies moments and mask", "[gpu][spectral_variance]")
{
  GpuSpectralVarianceHostResult host;
  host.sum_pow2 = {1.F, 2.F, 3.F, 4.F};
  host.sum_pow3 = {5.F, 6.F, 7.F, 8.F};
  host.sum_pow4 = {9.F, 10.F, 11.F, 12.F};
  host.nonzero_channels_mask = {1, 0, 1};

  const auto stats = assemble_spectral_variance_stats(host, 2, 2, 3);

  REQUIRE(stats.sum_pow2.shape(0) == 2);
  REQUIRE(stats.sum_pow2.shape(1) == 2);
  CHECK(stats.sum_pow2(0, 0) == 1.F);
  CHECK(stats.sum_pow2(0, 1) == 2.F);
  CHECK(stats.sum_pow2(1, 0) == 3.F);
  CHECK(stats.sum_pow2(1, 1) == 4.F);
  CHECK(stats.sum_pow3(1, 1) == 8.F);
  CHECK(stats.sum_pow4(1, 1) == 12.F);
  REQUIRE(stats.nonzero_channels_mask.shape(0) == 3);
  CHECK(stats.nonzero_channels_mask(0) == true);
  CHECK(stats.nonzero_channels_mask(1) == false);
  CHECK(stats.nonzero_channels_mask(2) == true);
  CHECK(stats.per_channels_squared.has_value() == false);
}

TEST_CASE("assemble_spectral_variance_stats validates sizes", "[gpu][spectral_variance]")
{
  GpuSpectralVarianceHostResult host;
  host.sum_pow2 = {1.F};
  host.sum_pow3 = {1.F};
  host.sum_pow4 = {1.F};
  host.nonzero_channels_mask = {1};

  CHECK_THROWS_AS(assemble_spectral_variance_stats(host, 2, 2, 1), std::runtime_error);
  CHECK_THROWS_AS(assemble_spectral_variance_stats(host, 0, 2, 1), std::runtime_error);
}
