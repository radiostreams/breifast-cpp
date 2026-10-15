#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpu/island_stamps.hpp"

using breifast::plane_lot::detail::stamp_beam_chi2;
using Catch::Approx;

TEST_CASE("stamp_beam_chi2 matches numpy on a seeded-random 5x5 stamp against a Gaussian kernel", "[stamp_beam_chi2]")
{
  // numpy (default_rng seed 11): stamp = normal(0, 1) + 40 * kernel in float32, then four pixels set to 0. The kernel is
  // a unit-sum Gaussian (sigma 1.2 by 0.9). 21 pixels stay nonzero. Reference in float64:
  //   scale = stamp.sum() / kernel.sum(); d = (stamp - scale * kernel)[stamp != 0]
  //   chi2 = (d ** 2).sum() / (1.3 ** 2 * 21)
  const std::vector<float> stamp = {
      0.0F,          0.0F,         2.7482831478118896F, 0.3115137219429016F, -0.16898779571056366F,
      -0.16185835003852844F, 2.898714065551758F, 4.261612892150879F, 3.0758731365203857F, -1.4817990064620972F,
      2.083815813064575F, 3.1993911266326904F, 6.7904582023620605F, 3.159256935119629F, 0.0F,
      0.8286359906196594F, 3.153501033782959F, 4.115147590637207F, 2.176201343536377F, 1.0512244701385498F,
      -0.7413589358329773F, -0.6925626993179321F, 1.918543815612793F, 0.0F, -1.7913588285446167F,
  };
  const std::vector<float> kernel = {
      0.003224542597308755F, 0.02054551988840103F, 0.038089048117399216F, 0.02054551988840103F, 0.003224542597308755F,
      0.009138146415352821F, 0.05822468921542168F, 0.10794193297624588F, 0.05822468921542168F, 0.009138146415352821F,
      0.012931677512824535F, 0.08239558339118958F, 0.15275199711322784F, 0.08239558339118958F, 0.012931677512824535F,
      0.009138146415352821F, 0.05822468921542168F, 0.10794193297624588F, 0.05822468921542168F, 0.009138146415352821F,
      0.003224542597308755F, 0.02054551988840103F, 0.038089048117399216F, 0.02054551988840103F, 0.003224542597308755F,
  };

  const float chi2 = stamp_beam_chi2(stamp, kernel, 2, 1.3F);

  CHECK(chi2 == Approx(0.5573078192176192).epsilon(1e-5));
}

TEST_CASE("stamp_beam_chi2 scales the kernel with the whole stamp and sums the residual over nonzero pixels only",
          "[stamp_beam_chi2]")
{
  // 3x3 stamp with one nonzero pixel (4) and a flat kernel of ones. numpy: scale = 4 / 9, the residual of the single
  // nonzero pixel is 4 - 4 / 9 = 32 / 9, chi2 = (32 / 9) ** 2 / (2 ** 2 * 1) = 3.16049...
  const std::vector<float> stamp = {
      0.F, 0.F, 0.F,
      0.F, 4.F, 0.F,
      0.F, 0.F, 0.F,
  };
  const std::vector<float> kernel(9, 1.F);

  CHECK(stamp_beam_chi2(stamp, kernel, 1, 2.F) == Approx(3.1604938271604937).epsilon(1e-6));
}

TEST_CASE("stamp_beam_chi2 is 0 for a perfect fit, an all-zero stamp and a zero adjacent noise", "[stamp_beam_chi2]")
{
  const std::vector<float> kernel(9, 1.F);

  // A flat stamp of 2 is exactly 2 * kernel: no residual.
  CHECK(stamp_beam_chi2(std::vector<float>(9, 2.F), kernel, 1, 1.5F) == 0.F);

  // No nonzero pixel: the chi2 guard returns 0.
  CHECK(stamp_beam_chi2(std::vector<float>(9, 0.F), kernel, 1, 1.5F) == 0.F);

  // A nonzero residual would exist, but adjacent_noise 0 returns 0 instead of a division by zero.
  const std::vector<float> stamp = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F, 9.F};
  CHECK(stamp_beam_chi2(stamp, kernel, 1, 0.F) == 0.F);
  CHECK(stamp_beam_chi2(stamp, kernel, 1, 1.F) > 0.F);

  // Radius 0: one pixel always fits itself.
  CHECK(stamp_beam_chi2(std::vector<float>{3.F}, std::vector<float>{0.5F}, 0, 1.F) == 0.F);
}

TEST_CASE("stamp_beam_chi2 rejects a negative halfsize and a stamp or kernel of the wrong size", "[stamp_beam_chi2]")
{
  const std::vector<float> nine(9, 1.F);
  using Catch::Matchers::ContainsSubstring;
  CHECK_THROWS_WITH(stamp_beam_chi2(nine, nine, -1, 1.F), ContainsSubstring("negative halfsize"));
  CHECK_THROWS_WITH(stamp_beam_chi2(std::vector<float>(8, 1.F), nine, 1, 1.F), ContainsSubstring("beam_stamp size"));
  CHECK_THROWS_WITH(stamp_beam_chi2(nine, std::vector<float>(10, 1.F), 1, 1.F), ContainsSubstring("beam_kernel size"));
  CHECK_THROWS_WITH(stamp_beam_chi2(nine, std::vector<float>(), 1, 1.F), ContainsSubstring("beam_kernel size"));
  CHECK_NOTHROW(stamp_beam_chi2(nine, nine, 1, 1.F));
}
