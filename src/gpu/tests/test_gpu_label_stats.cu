#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gpu_kernels.hpp"
#include "gpu_test_utils.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_label_stats;
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
  const auto device_count = upload(std::vector<unsigned int>{n_labels});
  DeviceVector<float> device_value(n_labels);
  DeviceVector<float> device_local_noise_max(n_labels);
  DeviceVector<long> device_peak_y(n_labels);
  DeviceVector<long> device_peak_x(n_labels);
  DeviceVector<unsigned int> device_island_size(n_labels);

  const Stream stream;
  gpu_label_stats(device_labels, device_image, device_local_noise, labels.size(), n_x, device_count, n_labels,
                  device_value, device_local_noise_max, device_peak_y, device_peak_x, device_island_size,
                  stream.get());
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

TEST_CASE("gpu_label_stats rejects an empty plane and an undersized capacity", "[gpu][label_stats]")
{
  using Catch::Matchers::ContainsSubstring;
  const Stream stream;

  DeviceVector<unsigned int> labels(1);
  DeviceVector<float> image(1);
  DeviceVector<float> local_noise(1);
  DeviceVector<float> value(1);
  DeviceVector<float> local_noise_max(1);
  DeviceVector<long> peak_y(1);
  DeviceVector<long> peak_x(1);
  DeviceVector<unsigned int> island_size(1);
  const auto count = upload(std::vector<unsigned int>{2});

  REQUIRE_THROWS_AS(gpu_label_stats(labels, image, local_noise, 0, 1, nullptr, 1, value, local_noise_max, peak_y,
                                    peak_x, island_size, stream.get()),
                    std::runtime_error);
  REQUIRE_THROWS_AS(gpu_label_stats(labels, image, local_noise, 1, 0, nullptr, 1, value, local_noise_max, peak_y,
                                    peak_x, island_size, stream.get()),
                    std::runtime_error);
  REQUIRE_THROWS_WITH(gpu_label_stats(labels, image, local_noise, 1, 1, count, 1, value, local_noise_max, peak_y,
                                      peak_x, island_size, stream.get()),
                      ContainsSubstring("exceeds capacity"));
}
