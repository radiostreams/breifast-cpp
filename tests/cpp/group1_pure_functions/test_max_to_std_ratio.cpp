#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "plane_lot_dispatch.hpp"

using namespace breifast;

namespace
{
float ratio_for_boxsize(long boxsize)
{
  PlaneConfig config;
  config.boxsize = boxsize;
  return config.max_to_std_ratio_value();
}
} // namespace

TEST_CASE("max_to_std_ratio_value matches an independent Python reference of the same algorithm",
         "[max_to_std_ratio]")
{
  // Reference values computed in Python, replicating the exact same 1000-point erf/interp search (including the
  // abs() breifast.py:73 applies to the interpolated crossing point).
  struct Case
  {
    long boxsize;
    float expected;
  };
  const std::vector<Case> cases = {
      {0, 0.000000F},   {1, 0.000000F},  {2, 1.445748F}, {3, 1.445748F}, {4, 1.921348F}, {5, 1.921348F},
      {10, 2.529462F}, {11, 2.529462F}, {20, 2.953590F}, {50, 3.463687F}, {100, 3.815543F},
  };
  for (const auto &c : cases)
  {
    CHECK(ratio_for_boxsize(c.boxsize) == Catch::Approx(c.expected).margin(0.01));
  }
}

TEST_CASE("max_to_std_ratio_value is 0 for a single-sample box (boxsize 0 or 1)", "[max_to_std_ratio]")
{
  // radius = boxsize/2 = 0 for both, giving n = 1: the median of a single standard normal sample is exactly 0.
  CHECK(ratio_for_boxsize(0) == Catch::Approx(0.0).margin(0.01));
  CHECK(ratio_for_boxsize(1) == Catch::Approx(0.0).margin(0.01));
}

TEST_CASE("max_to_std_ratio_value is always non-negative", "[max_to_std_ratio]")
{
  const std::vector<long> boxsizes = {0, 1, 2, 3, 4, 7, 10, 15, 20, 30, 50, 75, 100, 200, 500, 1000};
  for (const auto boxsize : boxsizes)
  {
    CHECK(ratio_for_boxsize(boxsize) >= 0.F);
  }
}

TEST_CASE("max_to_std_ratio_value is monotonically increasing in boxsize", "[max_to_std_ratio]")
{
  const std::vector<long> boxsizes = {0, 1, 2, 3, 4, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101, 151, 201};
  float previous = -1.F; // any value < 0, the ratio's known lower bound
  for (const auto boxsize : boxsizes)
  {
    const auto value = ratio_for_boxsize(boxsize);
    CHECK(value >= previous);
    previous = value;
  }
}

TEST_CASE("max_to_std_ratio_value is identical across separate PlaneConfig instances for the same boxsize",
         "[max_to_std_ratio]")
{
  const std::vector<long> boxsizes = {0, 3, 9, 21, 51, 101};
  for (const auto boxsize : boxsizes)
  {
    PlaneConfig config_a;
    config_a.boxsize = boxsize;
    PlaneConfig config_b;
    config_b.boxsize = boxsize;

    CHECK(config_a.max_to_std_ratio_value() == config_b.max_to_std_ratio_value());
  }
}

TEST_CASE("max_to_std_ratio_value is cached and stable across repeated calls on the same instance",
         "[max_to_std_ratio]")
{
  PlaneConfig config;
  config.boxsize = 17;

  const auto first = config.max_to_std_ratio_value();
  const auto second = config.max_to_std_ratio_value();
  const auto third = config.max_to_std_ratio_value();

  CHECK(first == second);
  CHECK(second == third);
}
