#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpu/islands.hpp"

using breifast::islands::island_boost;
using Catch::Approx;

namespace
{
// A 6x7 label image with three islands, and a baseline flux image, generated with numpy (default_rng seed 5: normal(0, 1)
// with the values below 0.3 in magnitude replaced by 0.5).
constexpr size_t n_y = 6;
constexpr size_t n_x = 7;

const std::vector<size_t> labels = {
    0, 1, 1, 1, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 2, 2,
    3, 0, 0, 0, 0, 0, 2,
    3, 3, 0, 0, 0, 0, 0,
};

const std::vector<float> baseline = {
    -0.8019314408302307F, -1.3243589401245117F, 0.5F,                 0.4204452335834503F, 1.1360465288162231F,
    0.5F,                 -0.5526472926139832F, -0.7847803831100464F, 0.7487457990646362F, 1.6347830295562744F,
    0.5F,                 -1.2333287000656128F, -0.9582651853561401F, 1.600019097328186F,  0.5F,
    -1.7321348190307617F, 0.5F,                 -1.1632260084152222F, -0.6292880773544312F, -0.4880058169364929F,
    -0.7133134007453918F, 0.5533784627914429F,  0.5F,                 -0.5894312858581543F, 0.40963783860206604F,
    0.8298553228378296F,  -1.6430233716964722F, 0.5F,                 -0.9807473421096802F, 0.5F,
    -1.2894186973571777F, 0.5F,                 0.5F,                 -0.304337739944458F,  -1.0479265451431274F,
    -0.396190345287323F,  -1.0913288593292236F, -1.3552087545394897F, 0.5F,                 -1.1093499660491943F,
    1.1702960729599F,     0.7165876626968384F,
};
} // namespace

TEST_CASE("island_boost matches numpy for three islands of a seeded-random baseline", "[island_boost]")
{
  // numpy, flux 7.5: r = flux / baseline[labels == L], boost = r[np.argmax(np.abs(r))].
  // Label 1: r = [-5.66, 15, 17.84, 4.59]     -> 17.838232040405273 (the largest, not the first).
  // Label 2: r = [-4.56, 15, -7.16]           -> 15.
  // Label 3: r = [-7.65, -18.93, -6.87]       -> -18.930294036865234 (a negative ratio wins on its magnitude).
  REQUIRE(labels.size() == n_y * n_x);
  REQUIRE(baseline.size() == n_y * n_x);

  CHECK(island_boost(labels, baseline, n_y, n_x, 1, 7.5F) == Approx(17.838232040405273).epsilon(1e-6));
  CHECK(island_boost(labels, baseline, n_y, n_x, 2, 7.5F) == Approx(15.0).epsilon(1e-6));
  CHECK(island_boost(labels, baseline, n_y, n_x, 3, 7.5F) == Approx(-18.930294036865234).epsilon(1e-6));
}

TEST_CASE("island_boost gives the first pixel in raster order when two ratios tie on magnitude", "[island_boost]")
{
  // flux 1 over baselines 0.5 and -0.5 gives ratios 2 and -2. numpy argmax of np.abs([2, -2]) is 0, then of
  // np.abs([-2, 2]) is 0 again: the first pixel wins and keeps its sign.
  const std::vector<size_t> two_pixels = {1, 1};
  CHECK(island_boost(two_pixels, std::vector<float>{0.5F, -0.5F}, 1, 2, 1, 1.F) == 2.F);
  CHECK(island_boost(two_pixels, std::vector<float>{-0.5F, 0.5F}, 1, 2, 1, 1.F) == -2.F);

  // The same tie across rows: raster order puts (0, 1) before (1, 0).
  const std::vector<size_t> diagonal = {0, 1, 1, 0};
  CHECK(island_boost(diagonal, std::vector<float>{9.F, -0.5F, 0.5F, 9.F}, 2, 2, 1, 1.F) == -2.F);
}

TEST_CASE("island_boost is the plain ratio for a one-pixel island and infinite where the baseline is 0",
          "[island_boost]")
{
  // numpy: 6 / 4 = 1.5 for the one-pixel island, other islands and background do not enter.
  const std::vector<size_t> image = {2, 0, 1, 0, 3, 3};
  const std::vector<float> flux_baseline = {100.F, 100.F, 4.F, 100.F, 100.F, 0.F};
  CHECK(island_boost(image, flux_baseline, 2, 3, 1, 6.F) == 1.5F);

  // Island 3 has a finite ratio (6 / 100) and a pixel with baseline 0. That pixel gives +inf or -inf by the sign of the
  // flux and wins the argmax.
  CHECK(island_boost(image, flux_baseline, 2, 3, 3, 6.F) == std::numeric_limits<float>::infinity());
  CHECK(island_boost(image, flux_baseline, 2, 3, 3, -6.F) == -std::numeric_limits<float>::infinity());
}

TEST_CASE("island_boost with a label that has no pixel throws", "[island_boost]")
{
  const std::vector<size_t> image = {1, 0, 1, 0};
  using Catch::Matchers::ContainsSubstring;
  CHECK_THROWS_WITH(island_boost(image, std::vector<float>{1.F, 1.F, 1.F, 1.F}, 2, 2, 2, 1.F),
                    ContainsSubstring("no pixel carries the label"));
}

TEST_CASE("island_boost rejects label and baseline images that do not hold n_y * n_x values", "[island_boost]")
{
  const std::vector<size_t> labels_2x2 = {1, 0, 1, 0};
  const std::vector<float> baseline_2x2 = {1.F, 1.F, 1.F, 1.F};
  using Catch::Matchers::ContainsSubstring;
  CHECK_THROWS_WITH(island_boost(std::vector<size_t>{1, 0, 1}, baseline_2x2, 2, 2, 1, 1.F),
                    ContainsSubstring("labels size"));
  CHECK_THROWS_WITH(island_boost(std::vector<size_t>(5, 1), baseline_2x2, 2, 2, 1, 1.F),
                    ContainsSubstring("labels size"));
  CHECK_THROWS_WITH(island_boost(labels_2x2, std::vector<float>{1.F, 1.F, 1.F}, 2, 2, 1, 1.F),
                    ContainsSubstring("baseline_flux size"));
  CHECK_THROWS_WITH(island_boost(labels_2x2, std::vector<float>(5, 1.F), 2, 2, 1, 1.F),
                    ContainsSubstring("baseline_flux size"));
  CHECK_NOTHROW(island_boost(labels_2x2, baseline_2x2, 2, 2, 1, 1.F));
}
