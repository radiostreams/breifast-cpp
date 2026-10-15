#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpu/island_stamps.hpp"

using breifast::plane_lot::detail::stamp_donut_stats;
using Catch::Approx;

TEST_CASE("stamp_donut_stats matches numpy on a seeded-random 9x9 stamp with a zeroed centre", "[stamp_donut_stats]")
{
  // numpy (default_rng seed 7, normal(0, 3), float32), then the centre 3x3 block and four more pixels set to 0.
  // 68 pixels stay nonzero. Reference: np.std(nz) and np.median(np.abs(nz)) * np.float32(1.4826).
  const std::vector<float> stamp = {
      0.0F,         0.8962366F,   -0.82241356F, -2.6717756F, -1.3640124F, -2.9749396F,  0.18043081F,  4.0206456F,   -1.4766196F,
      -1.8614247F,  1.4695262F,   1.0706611F,   0.31624275F, -2.7914042F, -0.087755464F, 2.0859096F,  -4.032644F,   -1.3728473F,
      -5.703668F,   -3.8686132F,  -5.525205F,   -0.7052734F, -3.8023396F, 0.81379306F,  0.47025326F,  0.0F,         -7.550279F,
      -1.6160787F,  -0.14550284F, 0.33992696F,  0.0F,        0.0F,        0.0F,         -2.4265118F,  3.1826959F,   -2.422604F,
      -0.097565114F, 2.6531696F,  -1.7508013F,  0.0F,        0.0F,        0.0F,         -3.6751676F,  0.22842069F,  4.0764704F,
      0.0F,         2.5781481F,   0.3580621F,   0.0F,        0.0F,        0.0F,         -3.5978668F,  0.22354868F,  1.7300688F,
      -0.56634635F, 2.0487309F,   -0.19955195F, 2.0017426F,  4.315568F,   -2.0269868F,  0.6094158F,   -1.3899227F,  0.38180524F,
      -3.5615835F,  -1.7379048F,  -0.58858794F, 2.6962917F,  3.435666F,   -3.9705834F,  -2.383927F,   1.9407103F,   -5.977259F,
      -1.3895096F,  -0.2918608F,  3.771045F,    0.0F,        -0.9816403F, -1.1057277F,  -0.7505862F,  4.570588F,    -1.2840748F,
  };
  REQUIRE(stamp.size() == 81);

  const auto stats = stamp_donut_stats(stamp, 4);

  CHECK(stats.stddev == Approx(2.610630869687416).epsilon(1e-6));
  CHECK(stats.lmad == Approx(2.5861778259277344F).epsilon(1e-6));
}

TEST_CASE("stamp_donut_stats averages the two middle ranks for an even count of nonzero pixels", "[stamp_donut_stats]")
{
  // 3x3 stamp with two nonzero pixels, -3 and 5. numpy: np.std([-3, 5]) = 4 (population), np.median([3, 5]) = 4, so
  // lmad is 4 * 1.4826. The sign of -3 matters for the spread and disappears in the absolute median.
  const std::vector<float> stamp = {
      0.F, -3.F, 0.F,
      0.F, 0.F,  0.F,
      5.F, 0.F,  0.F,
  };

  const auto stats = stamp_donut_stats(stamp, 1);

  CHECK(stats.stddev == Approx(4.0).epsilon(1e-12));
  CHECK(stats.lmad == Approx(4.F * 1.4826F).epsilon(1e-6));
}

TEST_CASE("stamp_donut_stats gives zeros for an all-zero stamp and a zero spread for one nonzero pixel",
          "[stamp_donut_stats]")
{
  // No donut pixel: both statistics are 0 (the empty-selection guard), for radius 0 and a larger stamp.
  const auto empty_small = stamp_donut_stats(std::vector<float>{0.F}, 0);
  CHECK(empty_small.stddev == 0.0);
  CHECK(empty_small.lmad == 0.F);

  const auto empty_large = stamp_donut_stats(std::vector<float>(25, 0.F), 2);
  CHECK(empty_large.stddev == 0.0);
  CHECK(empty_large.lmad == 0.F);

  // One nonzero pixel: np.std([-2.5]) = 0 and np.median([2.5]) * 1.4826 = 3.7065.
  std::vector<float> one(25, 0.F);
  one[7] = -2.5F;
  const auto single = stamp_donut_stats(one, 2);
  CHECK(single.stddev == 0.0);
  CHECK(single.lmad == Approx(3.7065F).epsilon(1e-6));
}

TEST_CASE("stamp_donut_stats rejects a negative radius and a stamp of the wrong size", "[stamp_donut_stats]")
{
  using Catch::Matchers::ContainsSubstring;
  CHECK_THROWS_WITH(stamp_donut_stats(std::vector<float>(9, 1.F), -1), ContainsSubstring("negative radius"));
  CHECK_THROWS_WITH(stamp_donut_stats(std::vector<float>(8, 1.F), 1), ContainsSubstring("size mismatch"));
  CHECK_THROWS_WITH(stamp_donut_stats(std::vector<float>(10, 1.F), 1), ContainsSubstring("size mismatch"));
  CHECK_THROWS_WITH(stamp_donut_stats(std::vector<float>(), 0), ContainsSubstring("size mismatch"));
  CHECK_NOTHROW(stamp_donut_stats(std::vector<float>(9, 1.F), 1));
}
