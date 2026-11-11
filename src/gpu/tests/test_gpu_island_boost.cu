#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gpu_band_workspace.cuh"
#include "gpu_kernels.hpp"
#include "gpu_test_utils.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_island_boost;
using breifast::gpu::IslandScratch;
using breifast::gpu::Stream;
using breifast::gpu::test::download_scalar;
using breifast::gpu::test::upload;

namespace
{
float run_boost(const std::vector<unsigned int> &labels, const std::vector<float> &baseline_flux,
                unsigned int label, float flux, IslandScratch &scratch)
{
  REQUIRE(labels.size() == baseline_flux.size());
  REQUIRE(label > 0U);

  const unsigned int active_count = labels.empty() ? 0U : *std::max_element(labels.begin(), labels.end());
  const unsigned int capacity = std::max(active_count, label);

  std::vector<float> host_flux(capacity, 0.F);
  host_flux[label - 1U] = flux;
  const auto device_labels = upload(labels);
  const auto device_baseline = upload(baseline_flux);
  const auto device_count = upload(std::vector<unsigned int>{active_count});
  const auto device_flux = upload(host_flux);
  DeviceVector<float> device_boost(capacity);
  DeviceVector<bool> device_found(capacity);
  scratch.ensure(capacity);

  const Stream stream;
  gpu_island_boost(device_labels, device_baseline, labels.size(), device_count, capacity, device_flux, device_boost,
                   device_found, scratch, stream.get());
  const float boost = download_scalar(device_boost.data() + (label - 1U), stream);
  const bool found = download_scalar(device_found.data() + (label - 1U), stream);

  if (!found)
  {
    throw std::runtime_error("no pixel carries the label");
  }
  return boost;
}

float run_boost(const std::vector<unsigned int> &labels, const std::vector<float> &baseline_flux,
                unsigned int label, float flux)
{
  IslandScratch scratch;
  return run_boost(labels, baseline_flux, label, flux, scratch);
}

float host_boost(const std::vector<unsigned int> &labels, const std::vector<float> &baseline_flux, unsigned int label,
                 float flux)
{
  bool found = false;
  float best_key = 0.F;
  float best_ratio = 0.F;
  size_t best_index = 0;
  for (size_t index = 0; index < labels.size(); ++index)
  {
    if (labels[index] != label)
    {
      continue;
    }

    const float ratio = flux / baseline_flux[index];
    const float key = std::fabs(ratio);
    if (!found || key > best_key || (key == best_key && index < best_index))
    {
      found = true;
      best_key = key;
      best_ratio = ratio;
      best_index = index;
    }
  }
  if (!found)
  {
    throw std::runtime_error("no pixel carries the label");
  }
  return best_ratio;
}
} // namespace

TEST_CASE("gpu_island_boost matches numpy for three islands of a seeded-random baseline", "[gpu][island_boost]")
{
  constexpr size_t n_y = 6;
  constexpr size_t n_x = 7;
  const std::vector<unsigned int> labels = {
      0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 2, 2, 3, 0, 0, 0, 0, 0, 2, 3, 3, 0, 0, 0, 0, 0,
  };
  const std::vector<float> baseline = {
      -0.8019314408302307F, -1.3243589401245117F, 0.5F,                0.4204452335834503F, 1.1360465288162231F,
      0.5F,                 -0.5526472926139832F, -0.7847803831100464F, 0.7487457990646362F, 1.6347830295562744F,
      0.5F,                 -1.2333287000656128F, -0.9582651853561401F, 1.600019097328186F,  0.5F,
      -1.7321348190307617F, 0.5F,                 -1.1632260084152222F, -0.6292880773544312F, -0.4880058169364929F,
      -0.7133134007453918F, 0.5533784627914429F,  0.5F,                 -0.5894312858581543F, 0.40963783860206604F,
      0.8298553228378296F,  -1.6430233716964722F, 0.5F,                 -0.9807473421096802F, 0.5F,
      -1.2894186973571777F, 0.5F,                 0.5F,                 -0.304337739944458F,  -1.0479265451431274F,
      -0.396190345287323F,  -1.0913288593292236F, -1.3552087545394897F, 0.5F,                 -1.1093499660491943F,
      1.1702960729599F,     0.7165876626968384F,
  };

  REQUIRE(labels.size() == n_y * n_x);
  REQUIRE(baseline.size() == n_y * n_x);

  CHECK_THAT(run_boost(labels, baseline, 1, 7.5F), Catch::Matchers::WithinRel(17.838232040405273, 1e-6));
  CHECK_THAT(run_boost(labels, baseline, 2, 7.5F), Catch::Matchers::WithinRel(15.0, 1e-6));
  CHECK_THAT(run_boost(labels, baseline, 3, 7.5F), Catch::Matchers::WithinRel(-18.930294036865234, 1e-6));
}

TEST_CASE("gpu_island_boost gives the first pixel in raster order when two ratios tie on magnitude",
          "[gpu][island_boost]")
{
  const std::vector<unsigned int> two_pixels = {1, 1};
  CHECK(run_boost(two_pixels, std::vector<float>{0.5F, -0.5F}, 1, 1.F) == 2.F);
  CHECK(run_boost(two_pixels, std::vector<float>{-0.5F, 0.5F}, 1, 1.F) == -2.F);

  const std::vector<unsigned int> diagonal = {0, 1, 1, 0};
  CHECK(run_boost(diagonal, std::vector<float>{9.F, -0.5F, 0.5F, 9.F}, 1, 1.F) == -2.F);
}

TEST_CASE("gpu_island_boost is the plain ratio for a one-pixel island and infinite where the baseline is 0",
          "[gpu][island_boost]")
{
  const std::vector<unsigned int> image = {2, 0, 1, 0, 3, 3};
  const std::vector<float> flux_baseline = {100.F, 100.F, 4.F, 100.F, 100.F, 0.F};
  CHECK(run_boost(image, flux_baseline, 1, 6.F) == 1.5F);
  CHECK(run_boost(image, flux_baseline, 3, 6.F) == std::numeric_limits<float>::infinity());
  CHECK(run_boost(image, flux_baseline, 3, -6.F) == -std::numeric_limits<float>::infinity());
}

TEST_CASE("gpu_island_boost with a label that has no pixel throws", "[gpu][island_boost]")
{
  const std::vector<unsigned int> image = {1, 0, 1, 0};
  using Catch::Matchers::ContainsSubstring;
  CHECK_THROWS_WITH(run_boost(image, std::vector<float>{1.F, 1.F, 1.F, 1.F}, 2, 1.F),
                    ContainsSubstring("no pixel carries the label"));
}

TEST_CASE("gpu_island_boost matches a host reference across many blocks", "[gpu][island_boost]")
{
  constexpr size_t n_y = 1024;
  constexpr size_t n_x = 1025;
  const size_t n = n_y * n_x;

  std::vector<unsigned int> labels(n);
  std::vector<float> baseline(n);
  for (size_t i = 0; i < n; ++i)
  {
    labels[i] = static_cast<unsigned int>((i * 7 + 3) % 5);
    baseline[i] = static_cast<float>((i * 13) % 17 + 1) * 0.5F;
  }

  for (unsigned int label = 1; label <= 4; ++label)
  {
    for (const float flux : {3.5F, -4.25F})
    {
      CHECK_THAT(run_boost(labels, baseline, label, flux),
                 Catch::Matchers::WithinRel(host_boost(labels, baseline, label, flux), 1e-6));
    }
  }
}

TEST_CASE("gpu_island_boost resets its keys when the scratch is reused", "[gpu][island_boost]")
{
  // The first call leaves label 1 a larger key (ratio 8 at pixel 0) than the second call produces (ratio 2 at pixel 1):
  // without the reset, the second call would keep pixel 0 and report 1 / 1.
  IslandScratch scratch;
  CHECK(run_boost({1, 1}, {0.125F, 1.F}, 1, 1.F, scratch) == 8.F);
  CHECK(run_boost({1, 1}, {1.F, 0.5F}, 1, 1.F, scratch) == 2.F);
}
