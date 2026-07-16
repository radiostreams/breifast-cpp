#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <array>
#include <cmath>
#include <random>

#include "../../src/breifast_core.hpp"

using Catch::Matchers::RangeEquals;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{

/**
 * @brief Builds a Catch2 RangeEquals matcher comparing elementwise with `|a - b| <= atol + rtol * |b|`.
 *
 * Lets array comparisons go through Catch2's own matcher/reporting machinery (`REQUIRE_THAT(actual,
 * AllClose(expected))`) instead of a hand-rolled pass/fail helper.
 *
 * \param expected Array to compare against.
 * \param atol Absolute tolerance.
 * \param rtol Relative tolerance, scaled by `expected`'s own magnitude.
 * \return A matcher usable with `REQUIRE_THAT`.
 */
auto AllClose(const xt::xarray<float> &expected, float atol = 1e-4f, float rtol = 1e-4f)
{
  return RangeEquals(expected, [atol, rtol](float a, float b) { return std::abs(a - b) <= atol + rtol * std::abs(b); });
}

/**
 * @brief Brute-force reference for the sliding-window RMS noise map.
 *
 * Independent of compute_integral_image()/box_sum() -- cross-checks make_noise_map()'s summed-area-table implementation
 * without relying on the same machinery it's meant to verify.
 *
 * \param plane (H, W) image to estimate noise for.
 * \param boxsize Side length of the sliding box.
 * \return (H, W) noise map, matching make_noise_map()'s own return convention.
 */
xt::xarray<float> brute_force_noise_map(const xt::xarray<float> &plane, long boxsize)
{
  const long h = static_cast<long>(plane.shape(0));
  const long w = static_cast<long>(plane.shape(1));
  const long lo = boxsize / 2;
  const double box_area = static_cast<double>(boxsize) * static_cast<double>(boxsize);

  xt::xarray<float> noise = xt::zeros<float>({static_cast<size_t>(h), static_cast<size_t>(w)});
  for (long i = 0; i < h; ++i)
  {
    const long r0 = std::max(0L, i - lo), r1 = std::min(h, i - lo + boxsize);
    for (long j = 0; j < w; ++j)
    {
      const long c0 = std::max(0L, j - lo), c1 = std::min(w, j - lo + boxsize);
      double sum2 = 0.0;
      long count = 0;
      for (long r = r0; r < r1; ++r)
      {
        for (long c = c0; c < c1; ++c)
        {
          const double v = static_cast<double>(plane(r, c));
          sum2 += v * v;
          ++count;
        }
      }
      const double scale = std::sqrt(static_cast<double>(count) / box_area);
      const double raw_noise = std::sqrt(sum2 / box_area);
      noise(i, j) = scale != 0.0 ? static_cast<float>(raw_noise / scale) : 0.f;
    }
  }
  const float med = xt::median(noise);
  xt::filter(noise, noise < med) = med;
  return noise;
}

/**
 * @brief Brute-force reference for a single rectangle sum, independent of compute_integral_image()'s
 * running-sum construction.
 *
 * \param img (H, W) image to sum over.
 * \param r0,r1 Row range [r0, r1), before clamping to the image bounds.
 * \param c0,c1 Column range [c0, c1), before clamping to the image bounds.
 * \return Sum over the clamped rectangle; 0 if it's empty after clamping.
 */
float brute_force_rect_sum(const xt::xarray<float> &img, long r0, long r1, long c0, long c1)
{
  const long h = static_cast<long>(img.shape(0));
  const long w = static_cast<long>(img.shape(1));
  r0 = std::clamp(r0, 0L, h);
  r1 = std::clamp(r1, 0L, h);
  c0 = std::clamp(c0, 0L, w);
  c1 = std::clamp(c1, 0L, w);
  double sum = 0.0;
  for (long r = r0; r < r1; ++r)
  {
    for (long c = c0; c < c1; ++c)
    {
      sum += static_cast<double>(img(r, c));
    }
  }
  return static_cast<float>(sum);
}

/**
 * @brief Generates an array of a given shape filled with values drawn from a normal distribution.
 *
 * \param rng Random number generator to draw from, advanced in place.
 * \param shape Array shape.
 * \param mean Mean of the normal distribution to draw values from.
 * \param stddev Standard deviation of the normal distribution to draw values from.
 * \return Array of dimensions `shape` filled with N(mean, stddev) samples.
 */
xt::xarray<float> random_array(std::mt19937 &rng, const std::vector<size_t> &shape, float mean, float stddev)
{
  std::normal_distribution<float> dist(mean, stddev);
  xt::xarray<float> arr(shape);
  for (auto &v : arr)
  {
    v = dist(rng);
  }
  return arr;
}

/**
 * @brief Generates a constant array with a given size and value, for xtensor validation purposes.
 *
 * \param shape Array shape.
 * \param value Value to fill the array with.
 * \return Array of dimensions `shape` filled with `value`.
 */
xt::xarray<float> constant_array(const std::vector<size_t> &shape, float value)
{
  xt::xarray<float> arr(shape);
  arr.fill(value);
  return arr;
}

} // namespace

TEST_CASE("compute_integral_image builds the expected summed-area table", "[compute_integral_image]")
{
  const xt::xarray<float> img{{1.f, 2.f}, {3.f, 4.f}};

  SECTION("Accum = double (production default)")
  {
    const xt::xarray<double> integral = compute_integral_image(img);
    const xt::xarray<double> expected{{0., 0., 0.}, {0., 1., 3.}, {0., 4., 10.}};
    REQUIRE(integral.shape() == expected.shape());
    REQUIRE_THAT(integral, RangeEquals(expected));
  }

  SECTION("Accum = float gives the same values for small integers")
  {
    const xt::xarray<float> integral = compute_integral_image<float>(img);
    REQUIRE(integral(2, 2) == 10.f);
    REQUIRE(integral(1, 1) == 1.f);
    REQUIRE(integral(2, 1) == 4.f);
  }

  SECTION("all-zero image gives an all-zero table")
  {
    const xt::xarray<float> zeros = xt::zeros<float>({3u, 3u});
    const xt::xarray<double> integral = compute_integral_image(zeros);
    for (auto v : integral)
    {
      REQUIRE(v == 0.);
    }
  }
}

TEST_CASE("box_sum computes clamped rectangle sums via inclusion-exclusion", "[box_sum]")
{
  const xt::xarray<float> img{{1.f, 2.f}, {3.f, 4.f}};
  const xt::xarray<double> integral = compute_integral_image(img);

  SECTION("full image")
  {
    REQUIRE(box_sum(integral, 0, 2, 0, 2, 2, 2) == 10.);
  }

  SECTION("single-pixel rectangles")
  {
    REQUIRE(box_sum(integral, 0, 1, 0, 1, 2, 2) == 1.);
    REQUIRE(box_sum(integral, 0, 1, 1, 2, 2, 2) == 2.);
    REQUIRE(box_sum(integral, 1, 2, 0, 1, 2, 2) == 3.);
    REQUIRE(box_sum(integral, 1, 2, 1, 2, 2, 2) == 4.);
  }

  SECTION("a row")
  {
    REQUIRE(box_sum(integral, 1, 2, 0, 2, 2, 2) == 7.);
  }

  SECTION("window extending before the image is clamped, not zero-padded away entirely")
  {
    REQUIRE(box_sum(integral, -5, 1, -5, 1, 2, 2) == 1.);
  }

  SECTION("window entirely past the image edge is empty")
  {
    REQUIRE(box_sum(integral, 5, 10, 0, 2, 2, 2) == 0.);
  }

  SECTION("inverted range (r0 >= r1) is empty")
  {
    REQUIRE(box_sum(integral, 1, 1, 0, 2, 2, 2) == 0.);
  }
}

TEST_CASE("box_sum matches a brute-force rectangle sum on random images", "[box_sum][compute_integral_image]")
{
  std::mt19937 rng(7);
  const xt::xarray<float> img = random_array(rng, {5, 7}, 0.f, 3.f);
  const xt::xarray<double> integral = compute_integral_image(img);

  // Windows that stay in-bounds, straddle an edge, or sit entirely outside the image.
  const std::vector<std::array<long, 4>> windows = {
      {0, 5, 0, 7}, {1, 3, 2, 6}, {-2, 3, -1, 4}, {3, 9, 4, 10}, {-10, -1, 0, 7}, {2, 2, 0, 7},
  };

  for (const auto &[r0, r1, c0, c1] : windows)
  {
    CAPTURE(r0, r1, c0, c1);
    const float expected = brute_force_rect_sum(img, r0, r1, c0, c1);
    const double actual = box_sum(integral, r0, r1, c0, c1, 5, 7);
    REQUIRE_THAT(actual, WithinRel(static_cast<double>(expected), 1e-5) || WithinAbs(0., 1e-9));
  }
}

TEST_CASE("make_noise_map on a constant plane returns the constant value everywhere", "[make_noise_map]")
{
  const xt::xarray<float> plane = constant_array({6u, 6u}, 2.5f);

  SECTION("Accum = double (production default)")
  {
    const auto [noise, mednoise] = make_noise_map(plane, 4);
    REQUIRE_THAT(noise, AllClose(constant_array({6u, 6u}, 2.5f)));
    REQUIRE_THAT(mednoise, WithinRel(2.5f));
  }

  SECTION("Accum = float")
  {
    const auto [noise, mednoise] = make_noise_map<float>(plane, 4);
    REQUIRE_THAT(noise, AllClose(constant_array({6u, 6u}, 2.5f)));
    REQUIRE_THAT(mednoise, WithinRel(2.5f));
  }
}

TEST_CASE("make_noise_map matches a brute-force sliding-window reference", "[make_noise_map]")
{
  std::mt19937 rng(11);
  const xt::xarray<float> plane = random_array(rng, {9, 11}, 0.f, 1.f);

  for (const long boxsize : {1L, 2L, 3L, 4L, 25L})
  {
    CAPTURE(boxsize);
    const auto [noise, mednoise] = make_noise_map(plane, boxsize);
    const xt::xarray<float> expected = brute_force_noise_map(plane, boxsize);
    REQUIRE_THAT(noise, AllClose(expected, 1e-3f, 1e-3f));
    REQUIRE_THAT(mednoise, WithinRel(static_cast<float>(xt::median(expected)), 1e-3f));
  }
}

TEST_CASE("make_noise_map<float> and make_noise_map<double> agree closely", "[make_noise_map]")
{
  std::mt19937 rng(13);
  const xt::xarray<float> plane = random_array(rng, {16, 16}, 0.f, 1e-4f);

  const auto [noise_f, mednoise_f] = make_noise_map<float>(plane, 8);
  const auto [noise_d, mednoise_d] = make_noise_map<double>(plane, 8);

  REQUIRE_THAT(noise_f, AllClose(noise_d, 1e-6f, 1e-3f));
  REQUIRE_THAT(mednoise_f, WithinRel(mednoise_d, 1e-3f));
}

TEST_CASE("histogram_counts bins masked values into unit-width sigma bins", "[histogram_counts]")
{
  // max_sigma_bin = 3 -> 6 bins covering [-3, -2), [-2, -1), ..., [2, 3].
  const xt::xarray<float> values{-3.5f, -3.0f, -2.5f, 0.0f, 2.9f, 3.0f, 3.5f};
  const xt::xarray<bool> mask{true, true, true, true, true, true, true};

  const std::vector<long> counts = histogram_counts(values, mask, 3);
  // -3.5 and 3.5 are outside [-3, 3] and dropped; -3.0 and -2.5 both land in bin 0; 0.0 -> bin 3;
  // 2.9 -> bin 5; 3.0 is in-range but its natural bin index (6) is clamped to the last bin (5).
  const std::vector<long> expected = {2, 0, 0, 1, 0, 2};
  REQUIRE_THAT(counts, RangeEquals(expected));
}

TEST_CASE("histogram_counts ignores masked-out pixels", "[histogram_counts]")
{
  const xt::xarray<float> values{-2.5f, -2.5f, 0.5f};
  const xt::xarray<bool> mask{true, false, true};

  const std::vector<long> counts = histogram_counts(values, mask, 3);
  const std::vector<long> expected = {1, 0, 0, 1, 0, 0};
  REQUIRE_THAT(counts, RangeEquals(expected));
}

TEST_CASE("histogram_counts returns all zeros when nothing is in range or masked in", "[histogram_counts]")
{
  const xt::xarray<float> values{100.f, -100.f};
  const xt::xarray<bool> mask{true, true};
  const std::vector<long> counts = histogram_counts(values, mask, 3);
  REQUIRE_THAT(counts, RangeEquals(std::vector<long>(6, 0)));

  const xt::xarray<float> values2{0.f, 1.f};
  const xt::xarray<bool> mask2{false, false};
  REQUIRE_THAT(histogram_counts(values2, mask2, 3), RangeEquals(std::vector<long>(6, 0)));
}

TEST_CASE("estimate_noise_and_snr with the flat MAD estimator", "[estimate_noise_and_snr]")
{
  const xt::xarray<float> plane_img{{1.f, -2.f}, {3.f, -4.f}};
  const xt::xarray<float> zeros = xt::zeros<float>({2u, 2u});
  const float noise_globmad = 2.0f;

  const NoiseMapResult result = estimate_noise_and_snr(plane_img, false, 0, noise_globmad, 1.f, zeros, zeros, 1.f);

  REQUIRE_THAT(result.noisemap, AllClose(constant_array({2u, 2u}, 2.0f)));
  REQUIRE_THAT(result.excess_noise_factor, AllClose(constant_array({2u, 2u}, 1.0f)));
  REQUIRE_THAT(result.mednoise, WithinRel(2.0f));
  REQUIRE_THAT(result.snr, AllClose(xt::xarray<float>{{0.5f, -1.f}, {1.5f, -2.f}}));
  REQUIRE_THAT(result.abs_snr, AllClose(xt::xarray<float>{{0.5f, 1.f}, {1.5f, 2.f}}));
  REQUIRE_THAT(result.valid_noisemap, RangeEquals(xt::xarray<bool>{{true, true}, {true, true}}));
}

TEST_CASE("estimate_noise_and_snr masks out pixels where the noise map is zero", "[estimate_noise_and_snr]")
{
  const xt::xarray<float> plane_img{{1.f, -2.f}, {3.f, -4.f}};
  const xt::xarray<float> zeros = xt::zeros<float>({2u, 2u});

  // noise_globmad = 0 and both extra variance terms zero -> noisemap0 = noisemap = 0 everywhere,
  // which must be treated as "invalid", not divided into.
  const NoiseMapResult result = estimate_noise_and_snr(plane_img, false, 0, 0.f, 1.f, zeros, zeros, 1.f);

  REQUIRE_THAT(result.noisemap, AllClose(zeros));
  REQUIRE_THAT(result.excess_noise_factor, AllClose(zeros));
  REQUIRE_THAT(result.snr, AllClose(zeros));
  REQUIRE_THAT(result.valid_noisemap, RangeEquals(xt::xarray<bool>{{false, false}, {false, false}}));
}

TEST_CASE("estimate_noise_and_snr with the sliding RMS estimator combines noisemap0 with the extra "
          "variance terms in quadrature",
          "[estimate_noise_and_snr][make_noise_map]")
{
  std::mt19937 rng(17);
  const xt::xarray<float> plane_img = random_array(rng, {4, 4}, 0.f, 1.f);
  const xt::xarray<float> selfnoise_offset = constant_array({4u, 4u}, 0.01f);
  const xt::xarray<float> dde_noise = constant_array({4u, 4u}, 0.02f);
  const float snsf = 1.5f;
  const float ddenoise_scaling_factor = 2.0f;
  const int boxsize = 2;

  const NoiseMapResult result = estimate_noise_and_snr(plane_img, true, boxsize, 999.f, snsf, selfnoise_offset,
                                                       dde_noise, ddenoise_scaling_factor);

  // Independently reproduce estimate_noise_and_snr()'s own combination formula from a fresh
  // make_noise_map() call, so this test exercises the combination logic itself rather than
  // make_noise_map() (covered separately above).
  const auto [noisemap0, unused_mednoise] = make_noise_map(plane_img, boxsize);
  (void)unused_mednoise;
  const xt::xarray<float> expected_noisemap =
      xt::sqrt(noisemap0 * noisemap0 + snsf * selfnoise_offset + ddenoise_scaling_factor * dde_noise);
  const xt::xarray<float> expected_excess = expected_noisemap / noisemap0;
  const xt::xarray<float> expected_snr = plane_img / expected_noisemap;

  REQUIRE_THAT(result.noisemap, AllClose(expected_noisemap));
  REQUIRE_THAT(result.excess_noise_factor, AllClose(expected_excess));
  REQUIRE_THAT(result.snr, AllClose(expected_snr));
  REQUIRE_THAT(result.valid_noisemap, RangeEquals(xt::xarray<bool>({4, 4}, true)));
  REQUIRE_THAT(result.mednoise, WithinRel(static_cast<float>(xt::median(expected_noisemap))));
}

TEST_CASE("apply_rejection_cutoffs narrows the mask by the SNR cutoff", "[apply_rejection_cutoffs]")
{
  const xt::xarray<bool> image_mask{{true, true}, {true, true}};
  const xt::xarray<float> abs_snr{{1.f, 2.f}, {3.f, 4.f}};
  const xt::xarray<float> snr = abs_snr;
  const xt::xarray<float> plane_img = xt::zeros<float>({2u, 2u});
  const xt::xarray<float> noisemap = xt::ones<float>({2u, 2u});

  // 2 channels; per-pixel band average of channel 0 (all 0) and channel 1 (all 4) is 2 everywhere.
  const xt::xarray<float> snr_cutoff{{{0.f, 0.f}, {0.f, 0.f}}, {{4.f, 4.f}, {4.f, 4.f}}};

  const CutoffResult result = apply_rejection_cutoffs(image_mask, snr, abs_snr, plane_img, noisemap, 1.f, snr_cutoff,
                                                      std::nullopt, 0.f, 0, 2, 2, 0, 3);

  REQUIRE_THAT(result.image_mask, RangeEquals(xt::xarray<bool>{{false, true}, {true, true}}));
  REQUIRE(result.hist1.empty());
  REQUIRE(result.hist2.empty());
}

TEST_CASE("apply_rejection_cutoffs narrows the mask by the flux cutoff", "[apply_rejection_cutoffs]")
{
  const xt::xarray<bool> image_mask{{true, true}, {true, true}};
  const xt::xarray<float> plane_img{{-1.f, -2.f}, {3.f, -5.f}};
  const xt::xarray<float> snr = xt::zeros<float>({2u, 2u});
  const xt::xarray<float> abs_snr = xt::zeros<float>({2u, 2u});
  const xt::xarray<float> noisemap = xt::ones<float>({2u, 2u});

  // Single channel, band average flux cutoff is 2 everywhere.
  const xt::xarray<float> flux_cutoff{{{2.f, 2.f}, {2.f, 2.f}}};

  const CutoffResult result = apply_rejection_cutoffs(image_mask, snr, abs_snr, plane_img, noisemap, 1.f, std::nullopt,
                                                      flux_cutoff, 0.f, 0, 1, 1, 0, 3);

  REQUIRE_THAT(result.image_mask, RangeEquals(xt::xarray<bool>{{false, true}, {true, true}}));
}

TEST_CASE("apply_rejection_cutoffs narrows the mask by the local-noise threshold", "[apply_rejection_cutoffs]")
{
  const xt::xarray<bool> image_mask{{true, true}, {true, true}};
  const xt::xarray<float> snr = xt::zeros<float>({2u, 2u});
  const xt::xarray<float> abs_snr = xt::zeros<float>({2u, 2u});
  const xt::xarray<float> plane_img = xt::zeros<float>({2u, 2u});
  const xt::xarray<float> noisemap{{1.f, 2.f}, {3.f, 10.f}};
  const float mednoise = 2.f;

  SECTION("nonzero threshold rejects pixels above mednoise * threshold")
  {
    const CutoffResult result = apply_rejection_cutoffs(image_mask, snr, abs_snr, plane_img, noisemap, mednoise,
                                                        std::nullopt, std::nullopt, 1.5f, 0, 1, 1, 0, 3);
    REQUIRE_THAT(result.image_mask, RangeEquals(xt::xarray<bool>{{true, true}, {true, false}}));
  }

  SECTION("threshold = 0 disables the cutoff entirely")
  {
    const CutoffResult result = apply_rejection_cutoffs(image_mask, snr, abs_snr, plane_img, noisemap, mednoise,
                                                        std::nullopt, std::nullopt, 0.f, 0, 1, 1, 0, 3);
    REQUIRE_THAT(result.image_mask, RangeEquals(image_mask));
  }
}

TEST_CASE("apply_rejection_cutoffs chains snr/flux cutoffs and reports hist1/hist2 for each stage",
          "[apply_rejection_cutoffs][histogram_counts]")
{
  const xt::xarray<bool> image_mask{{true, true}, {true, true}};
  const xt::xarray<float> snr{{-1.f, 2.f}, {-3.f, 4.f}};
  const xt::xarray<float> abs_snr{{1.f, 2.f}, {3.f, 4.f}};
  const xt::xarray<float> plane_img{{10.f, 10.f}, {10.f, 10.f}};
  const xt::xarray<float> noisemap = xt::ones<float>({2u, 2u});

  // snr_cutoff = 1.5 everywhere -> rejects the (0,0) pixel (abs_snr = 1 < 1.5).
  const xt::xarray<float> snr_cutoff{{{1.5f, 1.5f}, {1.5f, 1.5f}}};
  // flux_cutoff = 10 everywhere -> abs(plane_img) = 10 >= 10 passes for every pixel, so this
  // stage doesn't narrow the mask further.
  const xt::xarray<float> flux_cutoff{{{10.f, 10.f}, {10.f, 10.f}}};

  const CutoffResult result = apply_rejection_cutoffs(image_mask, snr, abs_snr, plane_img, noisemap, 1.f, snr_cutoff,
                                                      flux_cutoff, 0.f, 0, 1, 1, 2, 5);

  const xt::xarray<bool> mask_after_snr{{false, true}, {true, true}};
  const std::vector<long> expected_hist1 = histogram_counts(snr, mask_after_snr, 5);
  const std::vector<long> expected_hist2 = histogram_counts(snr, mask_after_snr, 5); // flux stage didn't narrow further

  REQUIRE_THAT(result.image_mask, RangeEquals(mask_after_snr));
  REQUIRE_THAT(result.hist1, RangeEquals(expected_hist1));
  REQUIRE_THAT(result.hist2, RangeEquals(expected_hist2));
}

TEST_CASE("apply_rejection_cutoffs skips hist1/hist2 depending on histogram_detail", "[apply_rejection_cutoffs]")
{
  const xt::xarray<bool> image_mask{{true, true}, {true, true}};
  const xt::xarray<float> snr{{-1.f, 2.f}, {-3.f, 4.f}};
  const xt::xarray<float> abs_snr = xt::abs(snr);
  const xt::xarray<float> plane_img = xt::zeros<float>({2u, 2u});
  const xt::xarray<float> noisemap = xt::ones<float>({2u, 2u});

  SECTION("histogram_detail = 0 computes neither")
  {
    const CutoffResult result = apply_rejection_cutoffs(image_mask, snr, abs_snr, plane_img, noisemap, 1.f,
                                                        std::nullopt, std::nullopt, 0.f, 0, 1, 1, 0, 5);
    REQUIRE(result.hist1.empty());
    REQUIRE(result.hist2.empty());
  }

  SECTION("histogram_detail = 1 computes only hist2")
  {
    const CutoffResult result = apply_rejection_cutoffs(image_mask, snr, abs_snr, plane_img, noisemap, 1.f,
                                                        std::nullopt, std::nullopt, 0.f, 0, 1, 1, 1, 5);
    REQUIRE(result.hist1.empty());
    REQUIRE_FALSE(result.hist2.empty());
  }

  SECTION("histogram_detail = 2 computes both")
  {
    const CutoffResult result = apply_rejection_cutoffs(image_mask, snr, abs_snr, plane_img, noisemap, 1.f,
                                                        std::nullopt, std::nullopt, 0.f, 0, 1, 1, 2, 5);
    REQUIRE_FALSE(result.hist1.empty());
    REQUIRE_FALSE(result.hist2.empty());
    REQUIRE_THAT(result.hist1,
                 RangeEquals(result.hist2)); // no cutoffs ran in between, so both stages see the same mask
  }
}
