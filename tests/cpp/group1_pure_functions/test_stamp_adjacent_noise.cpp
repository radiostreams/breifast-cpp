#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>

#include "cpu/island_stamps.hpp"

using breifast::plane_lot::detail::stamp_abs_extreme;
using breifast::plane_lot::detail::stamp_zero_inner_box;

namespace
{
// A 7x7 stamp (radius 3) holding 1 .. 49 in row-major order, like np.arange(1, 50).reshape(7, 7).
std::vector<float> counting_stamp()
{
  std::vector<float> stamp(49);
  for (size_t i = 0; i < stamp.size(); ++i)
  {
    stamp[i] = static_cast<float>(i + 1);
  }
  return stamp;
}
} // namespace

TEST_CASE("stamp_zero_inner_box zeros the centre block of a 7x7 stamp like numpy slice assignment",
          "[stamp_adjacent_noise]")
{
  // numpy: a[2:5, 2:5] = 0 for inner_boxsize 1, the 3x3 block around the centre pixel (3, 3).
  const std::vector<float> expected = {
    1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F,
    8.F, 9.F, 10.F, 11.F, 12.F, 13.F, 14.F,
    15.F, 16.F, 0.F, 0.F, 0.F, 20.F, 21.F,
    22.F, 23.F, 0.F, 0.F, 0.F, 27.F, 28.F,
    29.F, 30.F, 0.F, 0.F, 0.F, 34.F, 35.F,
    36.F, 37.F, 38.F, 39.F, 40.F, 41.F, 42.F,
    43.F, 44.F, 45.F, 46.F, 47.F, 48.F, 49.F,
  };
  auto stamp = counting_stamp();

  stamp_zero_inner_box(stamp, 3, 1);

  CHECK(stamp == expected);
}

TEST_CASE("stamp_zero_inner_box handles inner_boxsize 0, the radius itself and anything wider", "[stamp_adjacent_noise]")
{
  // Zero block of half-width 0: only the centre pixel (value 25).
  auto only_centre = counting_stamp();
  stamp_zero_inner_box(only_centre, 3, 0);
  auto expected_centre = counting_stamp();
  expected_centre[24] = 0.F;
  CHECK(only_centre == expected_centre);

  // Half-width equal to the radius, or wider (clipped to the stamp): the whole stamp.
  for (const long inner_boxsize : {3L, 4L, 100L})
  {
    INFO("inner_boxsize " << inner_boxsize);
    auto whole = counting_stamp();
    stamp_zero_inner_box(whole, 3, inner_boxsize);
    CHECK(whole == std::vector<float>(49, 0.F));
  }

  // Radius 0: a one-pixel stamp, zeroed by any inner_boxsize.
  std::vector<float> single = {7.F};
  stamp_zero_inner_box(single, 0, 0);
  CHECK(single == std::vector<float>{0.F});
}

TEST_CASE("stamp_zero_inner_box rejects a negative radius or inner_boxsize and a stamp of the wrong size",
          "[stamp_adjacent_noise]")
{
  auto stamp = counting_stamp();
  std::vector<float> short_stamp(48);

  using Catch::Matchers::ContainsSubstring;
  CHECK_THROWS_WITH(stamp_zero_inner_box(stamp, -1, 1), ContainsSubstring("negative radius"));
  CHECK_THROWS_WITH(stamp_zero_inner_box(stamp, 3, -1), ContainsSubstring("negative inner_boxsize"));
  CHECK_THROWS_WITH(stamp_zero_inner_box(short_stamp, 3, 1), ContainsSubstring("stamp size mismatch"));
  CHECK_THROWS_WITH(stamp_zero_inner_box(stamp, 2, 1), ContainsSubstring("stamp size mismatch"));
  CHECK_NOTHROW(stamp_zero_inner_box(stamp, 3, 1));
}

TEST_CASE("stamp_abs_extreme matches numpy's max(abs(min), abs(max)) on a seeded-random 5x5 stamp",
          "[stamp_adjacent_noise]")
{
  // numpy (default_rng seed 42, 2 * normal, float32): min -3.90207028, max 2.44508266, so the extreme is the minimum.
  const std::vector<float> stamp = {
      0.609434187F, -2.07996821F, 1.50090241F, 1.88112938F, -3.90207028F, -2.60435891F, 0.255680799F,
      -0.632485211F, -0.0336023159F, -1.70608783F, 1.75879598F, 1.55558383F, 0.132061392F, 2.25448251F,
      0.935018659F, -1.7185849F, 0.737501562F, -1.91776526F, 1.75690055F, -0.0998518243F, -0.369724721F,
      -1.36185908F, 2.44508266F, -0.309058964F, -0.856655657F,
  };

  CHECK(stamp_abs_extreme(stamp) == 3.90207028F);
}

TEST_CASE("stamp_abs_extreme takes the larger of the two sides and 0 for a zero stamp", "[stamp_adjacent_noise]")
{
  CHECK(stamp_abs_extreme(std::vector<float>{-5.F, 2.F, 3.F}) == 5.F);
  CHECK(stamp_abs_extreme(std::vector<float>{-2.F, 1.F, 4.F}) == 4.F);
  CHECK(stamp_abs_extreme(std::vector<float>{-4.F, 1.F, 4.F}) == 4.F);
  CHECK(stamp_abs_extreme(std::vector<float>{-3.F}) == 3.F);
  CHECK(stamp_abs_extreme(std::vector<float>(49, 0.F)) == 0.F);
  // All values of one sign: the other running extreme keeps its start value and does not matter.
  CHECK(stamp_abs_extreme(std::vector<float>{1.F, 2.F, 6.F}) == 6.F);
  CHECK(stamp_abs_extreme(std::vector<float>{-1.F, -2.F, -6.F}) == 6.F);
}

TEST_CASE("stamp_abs_extreme ignores NaN pixels and reports FLT_MAX when nothing else is there", "[stamp_adjacent_noise]")
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();

  CHECK(stamp_abs_extreme(std::vector<float>{nan, -7.F, 2.F, nan}) == 7.F);
  CHECK(stamp_abs_extreme(std::vector<float>{nan, nan}) == std::numeric_limits<float>::max());
  CHECK(stamp_abs_extreme(std::vector<float>{}) == std::numeric_limits<float>::max());
  CHECK(stamp_abs_extreme(std::vector<float>{1.F, -inf, 3.F}) == inf);
  CHECK(stamp_abs_extreme(std::vector<float>{1.F, inf, nan}) == inf);
}

TEST_CASE("stamp_abs_extreme agrees with the xtensor expression the detection loop used, NaN pixels included",
          "[stamp_adjacent_noise]")
{
  // The detection loop computed std::max(abs(minmax[0]), abs(minmax[1])) with xt::minmax. Random stamps of 7x7, with
  // about one pixel in ten NaN and the occasional infinity, must give the same value.
  std::mt19937 generator(42);
  std::normal_distribution<float> normal(0.F, 2.F);
  std::uniform_real_distribution<float> uniform(0.F, 1.F);
  for (int trial = 0; trial < 200; ++trial)
  {
    xt::xtensor<float, 2> stamp = xt::zeros<float>({7, 7});
    for (auto &value : stamp)
    {
      const float u = uniform(generator);
      value = u < 0.1F ? std::numeric_limits<float>::quiet_NaN()
              : u < 0.12F ? std::numeric_limits<float>::infinity()
                          : normal(generator);
    }
    const auto minmax = xt::minmax(stamp)();
    const float expected = std::max(std::abs(minmax[0]), std::abs(minmax[1]));

    INFO("trial " << trial);
    CHECK(stamp_abs_extreme({stamp.data(), stamp.size()}) == expected);
  }
}
