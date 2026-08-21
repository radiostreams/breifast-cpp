#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "plane_lot_dispatch.hpp"

using namespace breifast;

namespace
{
// Deterministic SplitMix64 generator, bit-for-bit identical to the Python reference script used to
// derive the expected values below -- avoids needing a multi-megabyte fixture file to reproduce a
// 3072x3072 plane.
xt::xarray<float> splitmix64_plane(size_t n_y, size_t n_x, uint64_t seed, float scale)
{
  xt::xarray<float> plane;
  plane.resize({n_y, n_x});
  uint64_t state = seed;
  for (size_t i_y = 0; i_y < n_y; ++i_y)
  {
    for (size_t i_x = 0; i_x < n_x; ++i_x)
    {
      state += 0x9E3779B97F4A7C15ULL;
      uint64_t z = state;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z = z ^ (z >> 31);
      const double u = static_cast<double>(z) / 18446744073709551616.0; // 2^64, exact in double
      plane(i_y, i_x) = static_cast<float>((u - 0.5) * 2.0 * static_cast<double>(scale));
    }
  }
  return plane;
}
} // namespace

TEST_CASE("compute_rms_noise_map matches a float64 Python reference on a large real-scale plane",
         "[compute_rms_noise_map]")
{
  // Real production configs use boxsize=400 on ~3072x3072 images. This regression-guards a numerical
  // precision bug found via real pipeline data: the summed-area table `compute_rms_noise_map` builds
  // internally used to sum in float32, and its recursive 2D formula compounds rounding error badly
  // enough over that many rows/columns to be off by several percent at some pixels
  const size_t n = 3072;
  const long boxsize = 400;
  const auto plane = splitmix64_plane(n, n, 0xC0FFEEULL, 0.001F);

  const auto noise_map = compute_rms_noise_map(plane, boxsize);

  // Expected values computed in Python
  struct ExpectedPoint
  {
    size_t y, x;
    double value;
  };
  const ExpectedPoint expected[] = {
      {2999, 2999, 0.0005774370365349552},
      {2500, 100, 0.0005774370365349552},
      {100, 2500, 0.000577586600176018},
      {1500, 1500, 0.0005785230252254532},
      {2999, 100, 0.0005774370365349552},
      {100, 2999, 0.0005786333131766458},
  };
  for (const auto &point : expected)
  {
    CHECK(noise_map(point.y, point.x) == Catch::Approx(point.value).epsilon(1e-5));
  }
}
