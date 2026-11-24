#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gpu_band.hpp"
#include "gpu_kernels.hpp"
#include "gpu_test_utils.hpp"

namespace label_islands_tests
{
using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_init_labels;
using breifast::gpu::gpu_label_islands;
using breifast::gpu::gpu_number_islands;
using breifast::gpu::LabelScratch;
using breifast::gpu::Stream;
using breifast::gpu::test::download;
using breifast::gpu::test::download_scalar;
using breifast::gpu::test::upload;
using Catch::Matchers::Equals;

namespace
{
std::vector<unsigned int> run_label(const std::vector<unsigned char> &mask, size_t n_y, size_t n_x,
                                    unsigned int *out_count = nullptr)
{
  const size_t n = mask.size();

  const auto device_mask = upload(std::vector<bool>(mask.begin(), mask.end()));
  DeviceVector<unsigned int> device_labels(n);
  DeviceVector<unsigned int> device_count(1);
  LabelScratch scratch;
  scratch.ensure(n);

  const Stream stream;
  gpu_init_labels(device_mask, n, device_labels, stream);
  gpu_label_islands(device_mask, n_y, n_x, device_labels, device_count, scratch, stream);
  gpu_number_islands(device_mask, n, device_labels, scratch, stream);
  const auto host_labels = download(device_labels, stream);
  const unsigned int count = download_scalar(device_count.data(), stream);

  if (out_count != nullptr)
  {
    *out_count = count;
  }
  return std::vector<unsigned int>(host_labels.begin(), host_labels.end());
}

std::vector<unsigned int> host_reference(const std::vector<unsigned char> &mask, size_t n_y, size_t n_x)
{
  const size_t n = n_y * n_x;
  std::vector<unsigned int> labels(n, 0);
  std::vector<size_t> stack;
  unsigned int next_label = 0;

  for (size_t start = 0; start < n; ++start)
  {
    if (mask[start] == 0 || labels[start] != 0)
    {
      continue;
    }

    ++next_label;
    labels[start] = next_label;
    stack.push_back(start);
    while (!stack.empty())
    {
      const size_t index = stack.back();
      stack.pop_back();
      const size_t i_y = index / n_x;
      const size_t i_x = index % n_x;

      const auto visit = [&](size_t neighbour)
      {
        if (mask[neighbour] != 0 && labels[neighbour] == 0)
        {
          labels[neighbour] = next_label;
          stack.push_back(neighbour);
        }
      };
      if (i_x > 0)
      {
        visit(index - 1);
      }
      if (i_x + 1 < n_x)
      {
        visit(index + 1);
      }
      if (i_y > 0)
      {
        visit(index - n_x);
      }
      if (i_y + 1 < n_y)
      {
        visit(index + n_x);
      }
    }
  }
  return labels;
}
} // namespace

TEST_CASE("gpu_label_islands numbers 4-connected islands in raster order of their first pixel", "[gpu][islands]")
{
  CHECK_THAT(run_label({1, 0, 1, 1, 1, 0, 0, 0, 1}, 3, 3), Equals(std::vector<unsigned int>{1, 0, 2, 1, 1, 0, 0, 0, 3}));
  CHECK_THAT(run_label({1, 0, 0, 1}, 1, 4), Equals(std::vector<unsigned int>{1, 0, 0, 2}));
  CHECK_THAT(run_label({0, 1, 0, 1, 1, 1, 0, 1, 0}, 3, 3),
             Equals(std::vector<unsigned int>{0, 1, 0, 1, 1, 1, 0, 1, 0}));
  CHECK_THAT(run_label({1, 1, 1, 1}, 2, 2), Equals(std::vector<unsigned int>{1, 1, 1, 1}));
}

TEST_CASE("gpu_label_islands keeps corner-touching cells separate under 4-connectivity", "[gpu][islands]")
{
  CHECK_THAT(run_label({1, 0, 0, 1}, 2, 2), Equals(std::vector<unsigned int>{1, 0, 0, 2}));
  CHECK_THAT(run_label({0, 1, 1, 0}, 2, 2), Equals(std::vector<unsigned int>{0, 1, 2, 0}));
  CHECK_THAT(run_label({1, 0, 1, 1}, 4, 1), Equals(std::vector<unsigned int>{1, 0, 2, 2}));
}

TEST_CASE("gpu_label_islands matches a host reference on a seeded multi-block mask", "[gpu][islands]")
{
  const size_t n_y = 37;
  const size_t n_x = 53;
  std::vector<unsigned char> mask(n_y * n_x);
  for (size_t i = 0; i < mask.size(); ++i)
  {
    mask[i] = static_cast<unsigned char>((i * 11 + 5) % 7 < 3);
  }

  unsigned int count = 0;
  CHECK_THAT(run_label(mask, n_y, n_x, &count), Equals(host_reference(mask, n_y, n_x)));
  CHECK(count > 1U);
}

TEST_CASE("gpu_label_islands propagates along a long thin island", "[gpu][islands]")
{
  const size_t n_x = 300;
  const std::vector<unsigned char> mask(n_x, 1);

  unsigned int count = 0;
  CHECK_THAT(run_label(mask, 1, n_x, &count), Equals(std::vector<unsigned int>(n_x, 1)));
  CHECK(count == 1U);
}

TEST_CASE("gpu_label_islands converges on a large multi-block mask", "[gpu][islands]")
{
  const size_t n_y = 256;
  const size_t n_x = 256;
  std::vector<unsigned char> mask(n_y * n_x, 0);
  for (size_t x = 0; x < n_x; ++x)
  {
    mask[x] = 1;
  }
  for (size_t y = 1; y < n_y; ++y)
  {
    for (size_t x = 0; x < n_x; x += 16)
    {
      mask[y * n_x + x] = 1;
    }
  }

  unsigned int count = 0;
  CHECK_THAT(run_label(mask, n_y, n_x, &count), Equals(host_reference(mask, n_y, n_x)));
  CHECK(count == 1U);
}

TEST_CASE("gpu_label_islands handles all-false and all-true masks", "[gpu][islands]")
{
  unsigned int count = 0;
  CHECK_THAT(run_label(std::vector<unsigned char>(6, 0), 2, 3, &count), Equals(std::vector<unsigned int>(6, 0)));
  CHECK(count == 0U);

  CHECK_THAT(run_label(std::vector<unsigned char>(6, 1), 2, 3, &count), Equals(std::vector<unsigned int>(6, 1)));
  CHECK(count == 1U);
}
} // namespace label_islands_tests

namespace label_stats_tests
{
using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_label_stats;
using breifast::gpu::IslandScratch;
using breifast::gpu::Stream;
using breifast::gpu::test::download;
using breifast::gpu::test::upload;
using Catch::Matchers::Equals;

namespace
{
struct LabelStats
{
  std::vector<float> value;
  std::vector<float> local_noise;
  std::vector<long> peak_y;
  std::vector<long> peak_x;
  std::vector<unsigned int> island_size;
};

LabelStats run_stats(const std::vector<unsigned int> &labels, const std::vector<float> &image,
                     const std::vector<float> &local_noise, size_t n_x, unsigned int n_labels)
{
  REQUIRE(labels.size() == image.size());
  REQUIRE(labels.size() == local_noise.size());

  const auto device_labels = upload(labels);
  const auto device_image = upload(image);
  const auto device_local_noise = upload(local_noise);
  DeviceVector<float> device_value(n_labels);
  DeviceVector<float> device_local_noise_max(n_labels);
  DeviceVector<long> device_peak_y(n_labels);
  DeviceVector<long> device_peak_x(n_labels);
  DeviceVector<unsigned int> device_island_size(n_labels);
  IslandScratch scratch;
  scratch.ensure(n_labels);

  const Stream stream;
  gpu_label_stats(device_labels, device_image, device_local_noise, labels.size(), n_x, n_labels, device_value,
                  device_local_noise_max, device_peak_y, device_peak_x, device_island_size, scratch, stream);
  return {download(device_value, stream), download(device_local_noise_max, stream), download(device_peak_y, stream),
          download(device_peak_x, stream), download(device_island_size, stream)};
}

LabelStats host_reference(const std::vector<unsigned int> &labels, const std::vector<float> &image,
                          const std::vector<float> &local_noise, size_t n_x, unsigned int n_labels)
{
  LabelStats stats;
  stats.value.assign(n_labels, -std::numeric_limits<float>::max());
  stats.local_noise.assign(n_labels, -std::numeric_limits<float>::max());
  stats.peak_y.assign(n_labels, 0);
  stats.peak_x.assign(n_labels, 0);
  stats.island_size.assign(n_labels, 0U);

  for (size_t index = 0; index < labels.size(); ++index)
  {
    const auto label = labels[index];
    if (label == 0U || label > n_labels)
    {
      continue;
    }

    const auto slot = label - 1U;
    ++stats.island_size[slot];

    const float noise = local_noise[index];
    if (!std::isnan(noise))
    {
      stats.local_noise[slot] = std::max(stats.local_noise[slot], noise);
    }

    const float abs_value = std::fabs(image[index]);
    if (!std::isnan(abs_value) && abs_value > stats.value[slot])
    {
      stats.value[slot] = abs_value;
      stats.peak_y[slot] = static_cast<long>(index / n_x);
      stats.peak_x[slot] = static_cast<long>(index % n_x);
    }
  }

  return stats;
}
} // namespace

TEST_CASE("gpu_label_stats reports size, peak, and local noise per island", "[gpu][label_stats]")
{
  constexpr size_t n_x = 7;
  const std::vector<unsigned int> labels = {
      0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 2, 2, 3, 0, 0, 0, 0, 0, 2, 3, 3, 0, 0, 0, 0, 0,
  };
  const std::vector<float> image = {
      -0.8F, -1.3F, 0.5F,  0.4F,  1.1F,  0.5F,  -0.5F, -0.7F, 0.7F,  1.6F,  0.5F,  -1.2F, -0.9F, 1.6F,
      0.5F,  -1.7F, 0.5F,  -1.1F, -0.6F, -0.4F, -0.7F, 0.5F,  0.5F,  -0.5F, 0.4F,  0.8F,  -1.6F, 0.5F,
      -0.9F, 0.5F,  -1.2F, 0.5F,  0.5F,  -0.3F, -1.0F, -0.3F, -1.0F, -1.3F, 0.5F,  -1.1F, 1.1F,  0.7F,
  };
  const std::vector<float> local_noise = {
      0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F, 0.9F, 1.0F, 1.1F, 1.2F, 1.3F, 1.4F,
      1.5F, 1.6F, 1.7F, 1.8F, 1.9F, 2.0F, 2.1F, 2.2F, 2.3F, 2.4F, 2.5F, 2.6F, 2.7F, 2.8F,
      2.9F, 3.0F, 3.1F, 3.2F, 3.3F, 3.4F, 3.5F, 3.6F, 3.7F, 3.8F, 3.9F, 4.0F, 4.1F, 4.2F,
  };

  const auto actual = run_stats(labels, image, local_noise, n_x, 3);
  const auto expected = host_reference(labels, image, local_noise, n_x, 3);

  CHECK_THAT(actual.value, Equals(expected.value));
  CHECK_THAT(actual.local_noise, Equals(expected.local_noise));
  CHECK_THAT(actual.peak_y, Equals(expected.peak_y));
  CHECK_THAT(actual.peak_x, Equals(expected.peak_x));
  CHECK_THAT(actual.island_size, Equals(expected.island_size));
}

TEST_CASE("gpu_label_stats keeps the first raster pixel on a magnitude tie", "[gpu][label_stats]")
{
  const std::vector<unsigned int> labels = {1, 1, 1, 1};
  const std::vector<float> image = {1.F, -2.F, 2.F, 1.F};
  const std::vector<float> local_noise = {1.F, 2.F, 3.F, 4.F};

  const auto actual = run_stats(labels, image, local_noise, 2, 1);
  const auto expected = host_reference(labels, image, local_noise, 2, 1);

  CHECK(actual.value[0] == 2.F);
  CHECK(actual.peak_y[0] == 0);
  CHECK(actual.peak_x[0] == 1);
  CHECK(actual.island_size[0] == 4U);
  CHECK(actual.local_noise[0] == 4.F);
  CHECK_THAT(actual.value, Equals(expected.value));
  CHECK_THAT(actual.peak_x, Equals(expected.peak_x));
}

TEST_CASE("gpu_label_stats leaves a label without pixels at its defaults", "[gpu][label_stats]")
{
  const std::vector<unsigned int> labels = {1, 0, 3, 0};
  const std::vector<float> image = {1.F, 2.F, 3.F, 4.F};
  const std::vector<float> local_noise = {1.F, 2.F, 3.F, 4.F};

  const auto actual = run_stats(labels, image, local_noise, 2, 3);
  const auto expected = host_reference(labels, image, local_noise, 2, 3);

  CHECK(actual.island_size[1] == 0U);
  CHECK(actual.value[1] == -std::numeric_limits<float>::max());
  CHECK(actual.local_noise[1] == -std::numeric_limits<float>::max());
  CHECK(actual.peak_y[1] == 0);
  CHECK(actual.peak_x[1] == 0);
  CHECK_THAT(actual.island_size, Equals(expected.island_size));
  CHECK_THAT(actual.value, Equals(expected.value));
  CHECK_THAT(actual.local_noise, Equals(expected.local_noise));
}

TEST_CASE("gpu_label_stats matches a host reference across many blocks", "[gpu][label_stats]")
{
  constexpr size_t n_x = 1025;
  constexpr size_t n_y = 1024;
  const size_t n = n_x * n_y;

  std::vector<unsigned int> labels(n);
  std::vector<float> image(n);
  std::vector<float> local_noise(n);
  for (size_t i = 0; i < n; ++i)
  {
    labels[i] = static_cast<unsigned int>((i * 7 + 3) % 5);
    image[i] = static_cast<float>((i * 13) % 17 - 8) * 0.5F;
    local_noise[i] = static_cast<float>((i * 11) % 19) * 0.25F;
  }

  const auto actual = run_stats(labels, image, local_noise, n_x, 4);
  const auto expected = host_reference(labels, image, local_noise, n_x, 4);

  CHECK_THAT(actual.value, Equals(expected.value));
  CHECK_THAT(actual.local_noise, Equals(expected.local_noise));
  CHECK_THAT(actual.peak_y, Equals(expected.peak_y));
  CHECK_THAT(actual.peak_x, Equals(expected.peak_x));
  CHECK_THAT(actual.island_size, Equals(expected.island_size));
}
} // namespace label_stats_tests

namespace island_boost_tests
{
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

  const unsigned int largest_label = labels.empty() ? 0U : *std::max_element(labels.begin(), labels.end());
  const unsigned int n_labels = std::max(largest_label, label);

  std::vector<float> host_flux(n_labels, 0.F);
  host_flux[label - 1U] = flux;
  const auto device_labels = upload(labels);
  const auto device_baseline = upload(baseline_flux);
  const auto device_flux = upload(host_flux);
  DeviceVector<float> device_boost(n_labels);
  scratch.ensure(n_labels);

  const Stream stream;
  gpu_island_boost(device_labels, device_baseline, labels.size(), n_labels, device_flux, device_boost, scratch, stream);
  return download_scalar(device_boost.data() + (label - 1U), stream);
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

TEST_CASE("gpu_island_boost gives a zero boost to a label with no pixel", "[gpu][island_boost]")
{
  // Label 2 is within the 3 labels but no pixel carries it.
  const std::vector<unsigned int> image = {1, 0, 3, 0};
  CHECK(run_boost(image, std::vector<float>{1.F, 1.F, 1.F, 1.F}, 2, 1.F) == 0.F);
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
} // namespace island_boost_tests
