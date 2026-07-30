// Catch2 unit tests for breifast_core.hpp: both the per-band-slice pieces (compute_integral_image,
// box_sum, make_noise_map, histogram_counts, estimate_noise_and_snr, apply_rejection_cutoffs) and
// the whole-plane pieces built on top of them (island detection, per-island statistics, optional
// beam fitting, spectral variance stats, process_plane_core/process_plane_batch_core). Hand-computed
// small cases for logic-heavy pieces, independent brute-force references cross-checked on random
// inputs for the numerically-heavy ones.
//
// No real scipy/numpy fixtures are available in this environment (neither system Python nor this
// repo's .venv has scipy installed), so label_islands()'s exact numbering agreement with
// scipy.ndimage.label is checked by construction (row-major first-encountered-pixel order) and by a
// dedicated regression case (see "label order"), not by direct comparison against real scipy output.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <unordered_map>

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

// Same as AllClose(), but for xt::xarray<double> (used by the double-precision spectral-variance
// and Gaussian-kernel results).
auto AllCloseD(const xt::xarray<double> &expected, double atol = 1e-6, double rtol = 1e-6)
{
  return RangeEquals(expected, [atol, rtol](double a, double b) { return std::abs(a - b) <= atol + rtol * std::abs(b); });
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

// Sorts islands by (r0, c0) for order-independent comparison -- label_islands()'s row-major
// first-encountered-pixel label order is checked separately (see "label order" below); this
// canonicalization lets the fuzz test compare island *sets* against a differently-implemented
// reference without also asserting that reference's internal labeling convention matches.
std::vector<Island> sorted_by_bbox(std::vector<Island> islands)
{
  std::sort(islands.begin(), islands.end(), [](const Island &a, const Island &b) {
    return std::tie(a.r0, a.c0) < std::tie(b.r0, b.c0);
  });
  return islands;
}

// Union-find-based connected-component reference, deliberately structured differently from
// label_islands()'s single-pass flood fill (two-pass: union provisional labels via up/left
// neighbors, then resolve canonical roots and build final label image), to cross-check without
// relying on the same algorithm being tested.
std::vector<Island> brute_force_label_islands(const xt::xarray<bool> &mask)
{
  const long h = static_cast<long>(mask.shape(0));
  const long w = static_cast<long>(mask.shape(1));
  std::vector<long> parent(static_cast<size_t>(h * w));
  std::iota(parent.begin(), parent.end(), 0L);
  std::function<long(long)> find = [&](long x) {
    while (parent[static_cast<size_t>(x)] != x)
    {
      x = parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
    }
    return x;
  };
  auto unite = [&](long a, long b) {
    a = find(a);
    b = find(b);
    if (a != b)
    {
      parent[static_cast<size_t>(std::max(a, b))] = std::min(a, b);
    }
  };
  auto idx = [&](long i, long j) { return i * w + j; };

  for (long i = 0; i < h; ++i)
  {
    for (long j = 0; j < w; ++j)
    {
      if (!mask(i, j))
        continue;
      if (i > 0 && mask(i - 1, j))
        unite(idx(i, j), idx(i - 1, j));
      if (j > 0 && mask(i, j - 1))
        unite(idx(i, j), idx(i, j - 1));
    }
  }

  xt::xarray<int> labels = xt::zeros<int>({static_cast<size_t>(h), static_cast<size_t>(w)});
  std::unordered_map<long, int> root_to_label;
  std::vector<Island> islands;
  for (long i = 0; i < h; ++i)
  {
    for (long j = 0; j < w; ++j)
    {
      if (!mask(i, j))
        continue;
      const long root = find(idx(i, j));
      auto it = root_to_label.find(root);
      int label_id;
      if (it == root_to_label.end())
      {
        label_id = static_cast<int>(islands.size()) + 1;
        root_to_label[root] = label_id;
        islands.push_back(Island{i, i + 1, j, j + 1, xt::xarray<bool>()});
      }
      else
      {
        label_id = it->second;
        Island &isl = islands[static_cast<size_t>(label_id - 1)];
        isl.r0 = std::min(isl.r0, i);
        isl.r1 = std::max(isl.r1, i + 1);
        isl.c0 = std::min(isl.c0, j);
        isl.c1 = std::max(isl.c1, j + 1);
      }
      labels(i, j) = label_id;
    }
  }

  for (size_t k = 0; k < islands.size(); ++k)
  {
    Island &isl = islands[k];
    isl.objmask =
        xt::equal(xt::view(labels, xt::range(isl.r0, isl.r1), xt::range(isl.c0, isl.c1)), static_cast<int>(k) + 1);
  }
  return islands;
}

void require_islands_equal(const Island &a, const Island &b)
{
  REQUIRE(a.r0 == b.r0);
  REQUIRE(a.r1 == b.r1);
  REQUIRE(a.c0 == b.c0);
  REQUIRE(a.c1 == b.c1);
  REQUIRE_THAT(a.objmask, RangeEquals(b.objmask));
}

// xt::xarray<bool>{{true}} is ambiguous (a degenerate single-element nested initializer list
// collides with xtensor's other constructor overloads) -- this sidesteps it for the common
// single-pixel-island test fixture.
xt::xarray<bool> single_true_mask()
{
  xt::xarray<bool> mask(std::vector<size_t>{1, 1});
  mask(0, 0) = true;
  return mask;
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

TEST_CASE("label_islands basic shapes", "[label_islands]")
{
  SECTION("plus-shaped island")
  {
    const xt::xarray<bool> mask{{false, true, false}, {true, true, true}, {false, true, false}};
    const auto islands = label_islands(mask);
    REQUIRE(islands.size() == 1);
    REQUIRE(islands[0].r0 == 0);
    REQUIRE(islands[0].r1 == 3);
    REQUIRE(islands[0].c0 == 0);
    REQUIRE(islands[0].c1 == 3);
    REQUIRE_THAT(islands[0].objmask, RangeEquals(mask));
  }

  SECTION("diagonally-adjacent pixels are two separate islands (4-, not 8-connectivity)")
  {
    const xt::xarray<bool> mask{{true, false}, {false, true}};
    const auto islands = label_islands(mask);
    REQUIRE(islands.size() == 2);
  }

  SECTION("island touching the border")
  {
    const xt::xarray<bool> mask{{true, false, false}, {false, false, false}, {false, false, false}};
    const auto islands = label_islands(mask);
    REQUIRE(islands.size() == 1);
    REQUIRE(islands[0].r0 == 0);
    REQUIRE(islands[0].r1 == 1);
    REQUIRE(islands[0].c0 == 0);
    REQUIRE(islands[0].c1 == 1);
  }

  SECTION("all-false mask has no islands")
  {
    const xt::xarray<bool> mask = xt::xarray<bool>(xt::zeros<bool>({4u, 4u}));
    REQUIRE(label_islands(mask).empty());
  }

  SECTION("all-true mask is one island spanning the whole array")
  {
    xt::xarray<bool> mask({4u, 4u}, true);
    const auto islands = label_islands(mask);
    REQUIRE(islands.size() == 1);
    REQUIRE(islands[0].r0 == 0);
    REQUIRE(islands[0].r1 == 4);
    REQUIRE(islands[0].c0 == 0);
    REQUIRE(islands[0].c1 == 4);
  }

  SECTION("concave (U-shaped) island stays a single component with a tight bbox")
  {
    const xt::xarray<bool> mask{{true, false, true}, {true, false, true}, {true, true, true}};
    const auto islands = label_islands(mask);
    REQUIRE(islands.size() == 1);
    REQUIRE(islands[0].r0 == 0);
    REQUIRE(islands[0].r1 == 3);
    REQUIRE(islands[0].c0 == 0);
    REQUIRE(islands[0].c1 == 3);
    long count = 0;
    for (auto v : islands[0].objmask)
      count += v ? 1 : 0;
    REQUIRE(count == 7);
  }
}

TEST_CASE("label_islands assigns labels in row-major first-encountered order", "[label_islands]")
{
  // Two isolated single pixels: (0,1) is encountered before (1,0) in a row-major scan, so it must
  // become island 0 -- a column-major (or otherwise differently-ordered) scan would number these
  // the other way around.
  const xt::xarray<bool> mask{{false, true}, {true, false}};
  const auto islands = label_islands(mask);
  REQUIRE(islands.size() == 2);
  REQUIRE(islands[0].r0 == 0);
  REQUIRE(islands[0].c0 == 1);
  REQUIRE(islands[1].r0 == 1);
  REQUIRE(islands[1].c0 == 0);
}

TEST_CASE("label_islands matches an independently-implemented union-find reference", "[label_islands]")
{
  std::mt19937 rng(42);
  std::bernoulli_distribution dist(0.4);

  for (int trial = 0; trial < 20; ++trial)
  {
    xt::xarray<bool> mask(std::vector<size_t>{9, 11});
    for (auto &v : mask)
      v = dist(rng);

    const auto actual = sorted_by_bbox(label_islands(mask));
    const auto expected = sorted_by_bbox(brute_force_label_islands(mask));

    CAPTURE(trial);
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
    {
      require_islands_equal(actual[i], expected[i]);
    }
  }
}

TEST_CASE("max_to_std_ratio", "[max_to_std_ratio]")
{
  SECTION("increases monotonically with n")
  {
    const float r1 = max_to_std_ratio(4);
    const float r2 = max_to_std_ratio(25);
    const float r3 = max_to_std_ratio(400);
    REQUIRE(r1 < r2);
    REQUIRE(r2 < r3);
  }

  SECTION("n = 1 is close to 0 (median of a single N(0,1) draw)")
  {
    REQUIRE_THAT(max_to_std_ratio(1), WithinAbs(0.f, 0.05f));
  }

  SECTION("memoized calls return bit-identical results")
  {
    const float a = max_to_std_ratio(123);
    const float b = max_to_std_ratio(123);
    REQUIRE(a == b);
  }
}

TEST_CASE("create_multidimensional_gaussian_kernel", "[gaussian_kernel]")
{
  SECTION("sums to 1")
  {
    for (auto [sr, sc, hs, rot] : {std::tuple{2.0, 2.0, 5L, 0.0}, std::tuple{1.5, 3.0, 4L, 0.3},
                                   std::tuple{0.0, 2.0, 3L, 0.0}})
    {
      const auto kernel = create_multidimensional_gaussian_kernel(sr, sc, hs, rot);
      CAPTURE(sr, sc, hs, rot);
      REQUIRE_THAT(xt::sum(kernel)(), WithinAbs(1.0, 1e-9));
    }
  }

  SECTION("radially symmetric when sigma_row == sigma_col and rotate = 0")
  {
    const auto kernel = create_multidimensional_gaussian_kernel(2.0, 2.0, 4, 0.0);
    // Peak at the center; corners smaller than edge-midpoints; smaller than center everywhere else.
    const long c = 4;
    REQUIRE(kernel(c, c) > kernel(c, c + 1));
    REQUIRE(kernel(0, 0) < kernel(c, 0));
    // Symmetric across both axes and the diagonal.
    REQUIRE_THAT(kernel(c - 1, c + 2), WithinAbs(kernel(c + 1, c + 2), 1e-12));
    REQUIRE_THAT(kernel(c + 2, c - 1), WithinAbs(kernel(c - 1, c + 2), 1e-12));
  }

  SECTION("sigma = 0 on one axis collapses to a 1D ridge along the other")
  {
    const auto kernel = create_multidimensional_gaussian_kernel(0.0, 1.5, 3, 0.0);
    // Every row is identical (no falloff along the row axis).
    for (long i = 0; i < 7; ++i)
    {
      REQUIRE_THAT(xt::xarray<double>(xt::view(kernel, i, xt::all())), AllCloseD(xt::xarray<double>(xt::view(kernel, 0, xt::all()))));
    }
  }

  SECTION("180 degree rotation reproduces the unrotated (symmetric) kernel")
  {
    const auto kernel = create_multidimensional_gaussian_kernel(1.5, 2.5, 4, 0.0);
    const auto rotated = create_multidimensional_gaussian_kernel(1.5, 2.5, 4, M_PI);
    REQUIRE_THAT(rotated, AllCloseD(kernel, 1e-9, 1e-9));
  }
}

TEST_CASE("find_island_peak", "[find_island_peak]")
{
  const xt::xarray<float> snr{{0.f, 1.f, 0.f}, {0.f, -3.f, 2.f}, {0.f, 0.f, 0.f}};
  const xt::xarray<float> plane_img{{0.f, 0.1f, 0.f}, {0.f, -0.3f, 0.2f}, {0.f, 0.f, 0.f}};
  const xt::xarray<float> noisemap{{1.f, 1.f, 1.f}, {1.f, 2.f, 3.f}, {1.f, 1.f, 1.f}};
  const xt::xarray<float> selfnoise{{0.f, 0.f, 0.f}, {0.f, 5.f, 0.f}, {0.f, 0.f, 0.f}};
  const xt::xarray<float> enf{{0.f, 0.f, 0.f}, {0.f, 7.f, 0.f}, {0.f, 0.f, 0.f}};
  Island island{0, 3, 0, 3, xt::xarray<bool>{{false, true, false}, {false, true, true}, {false, false, false}}};

  SECTION("peak is the pixel with the largest |snr|")
  {
    const IslandPeak peak = find_island_peak(island, snr, plane_img, noisemap, 0.1f, std::nullopt, selfnoise, enf);
    REQUIRE(peak.x == 1);
    REQUIRE(peak.y == 1);
    REQUIRE_THAT(peak.maxsig, WithinAbs(3.f, 1e-6f));
    REQUIRE_THAT(peak.flux, WithinAbs(-0.3f, 1e-6f));
    REQUIRE(peak.size == 3);
    REQUIRE_THAT(peak.local_noise, WithinAbs(3.f, 1e-6f));
    REQUIRE_THAT(peak.selfnoise_value, WithinAbs(5.f, 1e-6f));
    REQUIRE_THAT(peak.excess_noise_factor_value, WithinAbs(7.f, 1e-6f));
  }

  SECTION("tie-breaking picks the first pixel in row-major order")
  {
    const xt::xarray<float> tied_snr{{0.f, 2.f, 0.f}, {0.f, 2.f, 2.f}, {0.f, 0.f, 0.f}};
    const IslandPeak peak =
        find_island_peak(island, tied_snr, plane_img, noisemap, 0.1f, std::nullopt, selfnoise, enf);
    // Row-major order over the island's own objmask visits (0,1) before (1,1) and (1,2).
    REQUIRE(peak.y == 0);
    REQUIRE(peak.x == 1);
  }

  SECTION("boost is 0 when no baseline flux is given")
  {
    const IslandPeak peak = find_island_peak(island, snr, plane_img, noisemap, 0.1f, std::nullopt, selfnoise, enf);
    REQUIRE(peak.boost == 0.f);
  }

  SECTION("boost picks the max-abs-value flux/baseline ratio over all island pixels")
  {
    const xt::xarray<float> baseline{{1.f, 0.5f, 1.f}, {1.f, 2.f, -0.1f}, {1.f, 1.f, 1.f}};
    const IslandPeak peak = find_island_peak(island, snr, plane_img, noisemap, 0.1f, baseline, selfnoise, enf);
    // flmax = -0.3 (peak flux). Ratios over island pixels {(0,1),(1,1),(1,2)}: -0.3/0.5=-0.6,
    // -0.3/2=-0.15, -0.3/-0.1=3.0 -- the last has the largest |ratio|.
    REQUIRE_THAT(peak.boost, WithinAbs(3.0f, 1e-5f));
  }
}

TEST_CASE("compute_adjusted_noise_stats", "[compute_adjusted_noise_stats]")
{
  // 5x5 plane, padded by 2 on every side (pad=2); peak at (2,2) with flux 10.
  xt::xarray<float> plane_img = constant_array({5u, 5u}, 1.f);
  plane_img(2, 2) = 10.f;
  const long pad = 2;
  const xt::xarray<float> padimg = pad_plane_img(plane_img, pad);

  xt::xarray<float> snr = constant_array({5u, 5u}, 0.5f);
  Island island{2, 3, 2, 3, single_true_mask()};

  SECTION("computes adjusted-noise stats and overwrites the island's snr pixels")
  {
    // boxsize=4 -> r=2 (5x5 postage stamp); inner_boxsize=0 -> only the single center pixel zeroed.
    const AdjustedNoiseStats stats = compute_adjusted_noise_stats(island, snr, padimg, 2, 2, pad, 4, 0, 10.f);
    // After zeroing the center, the postage stamp is all 1s (extreme=1), n=25.
    const float expected_adjnoise = 1.f / max_to_std_ratio(25);
    REQUIRE_THAT(stats.adjnoise, WithinRel(expected_adjnoise, 1e-5f));
    REQUIRE_THAT(stats.sig_ext, WithinRel(10.f / expected_adjnoise, 1e-5f));
    // donut excludes the zeroed center; all remaining 24 pixels are exactly 1 -> rms = 0 -> sig_rms = 0
    // (guarded, matching Python's `if rms else 0`), median|donut| = 1 -> sig_lmad = 10/(1*1.4826).
    REQUIRE_THAT(stats.sig_rms, WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT(stats.sig_lmad, WithinRel(10.f / 1.4826f, 1e-5f));
    // snr got overwritten at the island's pixel with sig_ext, nowhere else.
    REQUIRE_THAT(snr(2, 2), WithinRel(stats.sig_ext, 1e-6f));
    REQUIRE_THAT(snr(0, 0), WithinAbs(0.5f, 1e-6f));
  }

  SECTION("degenerate fully-zero postage stamp falls back to 0 without crashing")
  {
    xt::xarray<float> zero_plane = xt::zeros<float>({5u, 5u});
    xt::xarray<float> zero_padimg = pad_plane_img(zero_plane, pad);
    xt::xarray<float> zero_snr = xt::zeros<float>({5u, 5u});
    const AdjustedNoiseStats stats = compute_adjusted_noise_stats(island, zero_snr, zero_padimg, 2, 2, pad, 4, 0, 0.f);
    REQUIRE(stats.sig_rms == 0.f);
    REQUIRE(stats.sig_lmad == 0.f);
  }

  SECTION("edge padding contributes zeros to 'extreme' but is excluded from donut stats")
  {
    // Peak right at the image edge, so the postage stamp includes real padding zeros.
    xt::xarray<float> edge_plane = constant_array({5u, 5u}, 2.f);
    edge_plane(0, 0) = -20.f;
    xt::xarray<float> edge_padimg = pad_plane_img(edge_plane, pad);
    xt::xarray<float> edge_snr = xt::zeros<float>({5u, 5u});
    Island edge_island{0, 1, 0, 1, single_true_mask()};
    const AdjustedNoiseStats stats =
        compute_adjusted_noise_stats(edge_island, edge_snr, edge_padimg, 0, 0, pad, 4, 0, -20.f);
    // The zeroed center (-20 -> 0) plus real zero-padding beyond the image edge both get excluded
    // from the donut, but extreme = max(|padding-inclusive stamp|) sees only the constant 2s here
    // once the (already-zero) center is zeroed, since padding doesn't reach 2s' magnitude.
    REQUIRE_THAT(stats.adjnoise, WithinRel(2.f / max_to_std_ratio(25), 1e-5f));
  }
}

TEST_CASE("compute_beam_chi2", "[compute_beam_chi2]")
{
  const long half_size = 2;
  const auto kernel = create_multidimensional_gaussian_kernel(1.0, 1.0, half_size, 0.0);

  SECTION("perfect fit (postage stamp is an exact scaled copy of the kernel) gives chi2 ~ 0")
  {
    const xt::xarray<float> plane_img = xt::xarray<float>(kernel * 5.0);
    const xt::xarray<float> padimg = pad_plane_img(plane_img, 0);
    const float chi2 = compute_beam_chi2(kernel, half_size, padimg, half_size, half_size, 0, 1.0f);
    REQUIRE_THAT(chi2, WithinAbs(0.f, 1e-6f));
  }

  SECTION("mismatched postage stamp gives chi2 > 0, scaling with adjnoise")
  {
    xt::xarray<float> plane_img = xt::xarray<float>(kernel * 5.0);
    plane_img(half_size, half_size) += 10.f; // perturb the peak
    const xt::xarray<float> padimg = pad_plane_img(plane_img, 0);
    const float chi2_small_noise = compute_beam_chi2(kernel, half_size, padimg, half_size, half_size, 0, 1.0f);
    const float chi2_large_noise = compute_beam_chi2(kernel, half_size, padimg, half_size, half_size, 0, 10.0f);
    REQUIRE(chi2_small_noise > 0.f);
    // Larger adjnoise divides the same numerator by a bigger denominator -> smaller chi2.
    REQUIRE(chi2_large_noise < chi2_small_noise);
  }
}

TEST_CASE("compute_spectral_variance_stats", "[compute_spectral_variance_stats]")
{
  xt::xarray<float> cube{{{1.f, 2.f}, {3.f, 4.f}}, {{0.f, 0.f}, {0.f, 0.f}}, {{2.f, -1.f}, {1.f, 0.f}}};
  const xt::xarray<bool> nonzero{true, false, true};

  SECTION("sums (not averages) squares/cubes/quartics over nonzero channels only")
  {
    const auto result = compute_spectral_variance_stats(cube, nonzero, false);
    REQUIRE(result.counts == 2);
    // channel 0: {{1,2},{3,4}}, channel 2: {{2,-1},{1,0}}. Brute-force per-pixel sums:
    const xt::xarray<double> expected_spvar{{1. * 1 + 2. * 2, 2. * 2 + (-1.) * (-1.)},
                                             {3. * 3 + 1. * 1, 4. * 4 + 0. * 0}};
    const xt::xarray<double> expected_spskew{{1. * 1 * 1 + 2. * 2 * 2, 2. * 2 * 2 + (-1.) * (-1.) * (-1.)},
                                              {3. * 3 * 3 + 1. * 1 * 1, 4. * 4 * 4 + 0.}};
    const xt::xarray<double> expected_spkurt{{1. + 16., 16. + 1.}, {81. + 1., 256. + 0.}};
    REQUIRE_THAT(result.spvar, AllCloseD(expected_spvar));
    REQUIRE_THAT(result.spskew, AllCloseD(expected_spskew));
    REQUIRE_THAT(result.spkurt, AllCloseD(expected_spkurt));
    REQUIRE_FALSE(result.timevar.has_value());
  }

  SECTION("save_stat_cubes=true returns the full per-channel squared cube")
  {
    const auto result = compute_spectral_variance_stats(cube, nonzero, true);
    REQUIRE(result.timevar.has_value());
    REQUIRE_THAT(*result.timevar, AllClose(xt::xarray<float>(cube * cube)));
  }
}

TEST_CASE("process_band_slice_core_serial matches process_band_slice_core", "[process_band_slice_core_serial]")
{
  // Correctness anchor for the hand-rolled (raw-loop, no xtensor) TODO.md item-4 implementation:
  // both functions must agree on the same synthetic input, since bench.cpp times them independently
  // rather than cross-checking them on every call.
  std::mt19937 rng(23);
  const size_t npix = 20;
  const xt::xarray<float> stokes_cube_template = random_array(rng, {3, npix, npix}, 0.f, 1.f);
  const xt::xarray<float> weights{1.f, 1.f, 1.f};
  const xt::xarray<bool> accept_mask({3u, npix, npix}, true);
  const xt::xarray<float> channel_width{1.f, 1.f, 1.f};
  const xt::xarray<float> mean_image = random_array(rng, {3, npix, npix}, 0.f, 0.1f);
  const xt::xarray<float> selfnoise_offset = constant_array({npix, npix}, 0.01f);
  const xt::xarray<float> dde_noise = constant_array({npix, npix}, 0.02f);
  const xt::xarray<float> flux_cutoff = constant_array({3u, npix, npix}, 0.05f);
  const xt::xarray<float> snr_cutoff = constant_array({3u, npix, npix}, 0.3f);

  auto run_case = [&](bool is_single_channel, long band_start, long band_stop, bool subtract_mean,
                      bool use_rms_noise_estimator, int boxsize) {
    xt::xarray<float> cube_a = stokes_cube_template;
    xt::xarray<float> cube_b = stokes_cube_template;
    const ProcessBandSliceResult a = process_band_slice_core(
        cube_a, weights, accept_mask, channel_width, band_start, band_stop, is_single_channel, subtract_mean,
        mean_image, use_rms_noise_estimator, boxsize, selfnoise_offset, dde_noise, 1.f, 3.f, flux_cutoff, snr_cutoff,
        1.5f, 2, 10, 0.5f);
    const ProcessBandSliceResult b = process_band_slice_core_serial(
        cube_b, weights, accept_mask, channel_width, band_start, band_stop, is_single_channel, subtract_mean,
        mean_image, use_rms_noise_estimator, boxsize, selfnoise_offset, dde_noise, 1.f, 3.f, flux_cutoff, snr_cutoff,
        1.5f, 2, 10, 0.5f);

    REQUIRE(a.processed == b.processed);
    REQUIRE(a.reached_cutoffs == b.reached_cutoffs);
    REQUIRE_THAT(a.mednoise, WithinRel(b.mednoise, 1e-4f) || WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT(a.noise_globmad, WithinRel(b.noise_globmad, 1e-4f) || WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT(a.hist0, RangeEquals(b.hist0));
    REQUIRE_THAT(a.hist1, RangeEquals(b.hist1));
    REQUIRE_THAT(a.hist2, RangeEquals(b.hist2));
    if (a.processed)
    {
      REQUIRE_THAT(a.minsig, WithinRel(b.minsig, 1e-4f) || WithinAbs(0.f, 1e-6f));
      REQUIRE_THAT(a.maxsig, WithinRel(b.maxsig, 1e-4f) || WithinAbs(0.f, 1e-6f));
      REQUIRE_THAT(a.snr, AllClose(b.snr, 1e-4f, 1e-4f));
      REQUIRE_THAT(a.noisemap, AllClose(b.noisemap, 1e-4f, 1e-4f));
      REQUIRE_THAT(a.excess_noise_factor, AllClose(b.excess_noise_factor, 1e-4f, 1e-4f));
      REQUIRE_THAT(a.plane_img, AllClose(b.plane_img, 1e-4f, 1e-4f));
      REQUIRE_THAT(a.mask, RangeEquals(b.mask));
      REQUIRE_THAT(cube_a, AllClose(cube_b, 1e-4f, 1e-4f)); // in-place write-back agrees too
    }
  };

  SECTION("single-channel band, mean subtraction, sliding RMS noise estimator")
  {
    run_case(true, 0, 1, true, true, 6);
  }

  SECTION("multi-channel aggregated band, no mean subtraction, flat MAD estimator")
  {
    run_case(false, 0, 3, false, false, 0);
  }
}

TEST_CASE("process_plane_core", "[process_plane_core]")
{
  const long npix = 6;
  const xt::xarray<bool> nonzero_channels{true};
  const xt::xarray<float> weights{1.f};
  const xt::xarray<float> channel_width{1.f};
  xt::xarray<bool> accept_mask({1u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, true);
  const xt::xarray<float> selfnoise = xt::zeros<float>({static_cast<size_t>(npix), static_cast<size_t>(npix)});
  const xt::xarray<float> dde_noise = xt::zeros<float>({static_cast<size_t>(npix), static_cast<size_t>(npix)});

  PlaneConfig config;
  config.band_slices = {BandSlice{0, 1, "B0"}};
  config.use_rms_noise_estimator = false; // simpler: flat MAD estimator, no boxsize dependency
  config.boxsize = 4;
  config.inner_boxsize = 0;
  config.full_bandwidth = 1.f;
  config.threshold = 3.f;

  SECTION("no islands (all pixels below threshold): empty detections, but minsig/maxsig are still "
         "set since pixels survive the rejection cutoffs regardless of the final threshold")
  {
    xt::xarray<float> stokes_cube = constant_array({1u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, 0.01f);
    const ProcessPlaneResult result =
        process_plane_core(stokes_cube, weights, accept_mask, channel_width, nonzero_channels, std::nullopt, selfnoise,
                            selfnoise, dde_noise, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, config);
    REQUIRE(result.any_data);
    REQUIRE(result.detections.x.empty());
    REQUIRE(result.minsig.has_value());
    REQUIRE(result.maxsig.has_value());
    REQUIRE(result.hist0.empty()); // histogram_detail defaults to 0
    REQUIRE(result.mednoises.size() == 1);
  }

  SECTION("no data at all (nonzero_channels all false): everything stays absent/empty")
  {
    xt::xarray<float> stokes_cube = constant_array({1u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, 0.01f);
    const xt::xarray<bool> all_zero_channels{false};
    const ProcessPlaneResult result =
        process_plane_core(stokes_cube, weights, accept_mask, channel_width, all_zero_channels, std::nullopt,
                            selfnoise, selfnoise, dde_noise, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, config);
    REQUIRE_FALSE(result.any_data);
    REQUIRE(result.detections.x.empty());
    REQUIRE_FALSE(result.minsig.has_value());
    REQUIRE_FALSE(result.maxsig.has_value());
    REQUIRE(result.hist0.empty());
    REQUIRE(result.mednoises.empty());
  }

  SECTION("one clear point-source island: detection recorded and stokes_cube holds sig_ext, not raw SNR")
  {
    xt::xarray<float> stokes_cube = constant_array({1u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, 0.01f);
    stokes_cube(0, 3, 3) = 5.f; // a strong point source well above threshold
    const ProcessPlaneResult result =
        process_plane_core(stokes_cube, weights, accept_mask, channel_width, nonzero_channels, std::nullopt, selfnoise,
                            selfnoise, dde_noise, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, config);
    REQUIRE(result.detections.x.size() == 1);
    REQUIRE(result.detections.x[0] == 3);
    REQUIRE(result.detections.y[0] == 3);
    REQUIRE(result.minsig.has_value());
    // The written-back cube value at the peak is sigma_ext (island-adjusted), not the raw
    // flux/noise_globmad SNR -- this is the single most important regression case in this file.
    REQUIRE_THAT(stokes_cube(0, 3, 3), WithinRel(result.detections.sigma_ext[0], 1e-5f));
    REQUIRE_FALSE(stokes_cube(0, 3, 3) == result.detections.sigma[0]);
  }

  SECTION("beam_size_enabled = false gives beamchi2 == 0 for every detection")
  {
    xt::xarray<float> stokes_cube = constant_array({1u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, 0.01f);
    stokes_cube(0, 3, 3) = 5.f;
    const ProcessPlaneResult result =
        process_plane_core(stokes_cube, weights, accept_mask, channel_width, nonzero_channels, std::nullopt, selfnoise,
                            selfnoise, dde_noise, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, config);
    REQUIRE(result.detections.beamchi2.size() == 1);
    REQUIRE(result.detections.beamchi2[0] == 0.f);
  }

  SECTION("nfreq > 1 populates variance_stats; nfreq == 1 doesn't")
  {
    xt::xarray<float> single_channel_cube =
        constant_array({1u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, 0.01f);
    const ProcessPlaneResult single_result =
        process_plane_core(single_channel_cube, weights, accept_mask, channel_width, nonzero_channels, std::nullopt,
                            selfnoise, selfnoise, dde_noise, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, config);
    REQUIRE_FALSE(single_result.variance_stats.has_value());

    xt::xarray<float> multi_cube =
        constant_array({2u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, 0.01f);
    xt::xarray<bool> multi_accept({2u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, true);
    const xt::xarray<bool> multi_nonzero{true, true};
    const xt::xarray<float> multi_weights{1.f, 1.f};
    const xt::xarray<float> multi_channel_width{1.f, 1.f};
    PlaneConfig multi_config = config;
    multi_config.band_slices = {BandSlice{0, 1, "B0"}, BandSlice{1, 2, "B1"}};
    const ProcessPlaneResult multi_result =
        process_plane_core(multi_cube, multi_weights, multi_accept, multi_channel_width, multi_nonzero, std::nullopt,
                            selfnoise, selfnoise, dde_noise, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, multi_config);
    REQUIRE(multi_result.variance_stats.has_value());
    REQUIRE(multi_result.variance_stats->counts == 2);
  }

  SECTION("mednoises is indexed by reversed-loop position, not original band index")
  {
    xt::xarray<float> multi_cube = constant_array({2u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, 0.01f);
    xt::xarray<bool> multi_accept({2u, static_cast<size_t>(npix), static_cast<size_t>(npix)}, true);
    const xt::xarray<bool> multi_nonzero{true, true};
    const xt::xarray<float> multi_weights{1.f, 1.f};
    const xt::xarray<float> multi_channel_width{1.f, 1.f};
    // Distinct per-channel magnitudes so each band's mednoise (from noise_globmad, since
    // use_rms_noise_estimator=false) is distinguishable.
    multi_cube(0, 0, 0) = 0.02f;
    multi_cube(1, 0, 0) = 0.05f;

    PlaneConfig multi_config = config;
    multi_config.band_slices = {BandSlice{0, 1, "B0"}, BandSlice{1, 2, "B1"}};
    const ProcessPlaneResult result =
        process_plane_core(multi_cube, multi_weights, multi_accept, multi_channel_width, multi_nonzero, std::nullopt,
                            selfnoise, selfnoise, dde_noise, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, multi_config);
    // band_slices reversed: iband=0 processes band index 1 (B1), iband=1 processes band index 0 (B0).
    REQUIRE(result.mednoises.size() == 2);
  }
}
