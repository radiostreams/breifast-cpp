#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gpu_buffers.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_runtime.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_island_boost;
using breifast::gpu::PinnedVector;

namespace
{
float run_boost(const std::vector<unsigned int> &labels, const std::vector<float> &baseline_flux,
                unsigned int label, float flux)
{
  REQUIRE(labels.size() == baseline_flux.size());

  PinnedVector<unsigned int> host_labels(labels.begin(), labels.end());
  PinnedVector<float> host_baseline(baseline_flux.begin(), baseline_flux.end());
  PinnedVector<unsigned int> host_labels_to_process(1, label);
  PinnedVector<float> host_flux_per_label(1, flux);
  DeviceVector<unsigned int> device_labels(labels.size());
  DeviceVector<float> device_baseline(baseline_flux.size());
  DeviceVector<unsigned int> device_labels_to_process(1);
  DeviceVector<float> device_flux_per_label(1);
  DeviceVector<float> device_boost(1);
  DeviceVector<unsigned char> device_found(1);

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_labels_to_process.data()),
                             thrust::raw_pointer_cast(host_labels_to_process.data()), sizeof(unsigned int), stream);
  gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_flux_per_label.data()),
                             thrust::raw_pointer_cast(host_flux_per_label.data()), sizeof(float), stream);
  if (!labels.empty())
  {
    gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_labels.data()),
                               thrust::raw_pointer_cast(host_labels.data()), labels.size() * sizeof(unsigned int),
                               stream);
    gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_baseline.data()),
                               thrust::raw_pointer_cast(host_baseline.data()), baseline_flux.size() * sizeof(float),
                               stream);
  }

  gpu_island_boost(thrust::raw_pointer_cast(device_labels.data()), labels.size(), 1,
                   thrust::raw_pointer_cast(device_baseline.data()),
                   thrust::raw_pointer_cast(device_labels_to_process.data()),
                   thrust::raw_pointer_cast(device_flux_per_label.data()),
                   thrust::raw_pointer_cast(device_boost.data()),
                   reinterpret_cast<bool *>(thrust::raw_pointer_cast(device_found.data())), stream);

  float boost = 0.F;
  unsigned char found = 0;
  gpuMemcpyDeviceToHostAsync(&boost, thrust::raw_pointer_cast(device_boost.data()), sizeof(float), stream);
  gpuMemcpyDeviceToHostAsync(&found, thrust::raw_pointer_cast(device_found.data()), sizeof(unsigned char), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  if (found == 0)
  {
    throw std::runtime_error("no pixel carries the label");
  }
  return boost;
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

TEST_CASE("gpu_island_boost rejects an empty plane", "[gpu][island_boost]")
{
  using Catch::Matchers::ContainsSubstring;
  CHECK_THROWS_WITH(run_boost({}, {}, 1, 1.F), ContainsSubstring("greater than 0"));
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
