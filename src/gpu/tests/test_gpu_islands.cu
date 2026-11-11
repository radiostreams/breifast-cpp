#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "device_vector.hpp"
#include "gpu_kernels.hpp"
#include "gpu_runtime.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_label_islands;
using Catch::Matchers::Equals;

namespace
{
std::vector<unsigned int> run_label(const std::vector<unsigned char> &mask, size_t n_y, size_t n_x,
                                    unsigned int *out_count = nullptr)
{
  const size_t n = mask.size();

  std::vector<unsigned char> host_mask(mask.begin(), mask.end());
  DeviceVector<unsigned char> device_mask(n);
  DeviceVector<unsigned int> device_labels(n);
  DeviceVector<unsigned int> device_count(1);
  DeviceVector<unsigned int> device_scratch(n);

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_mask.data()), thrust::raw_pointer_cast(host_mask.data()),
                             n * sizeof(unsigned char), stream);
  gpu_label_islands(reinterpret_cast<const bool *>(thrust::raw_pointer_cast(device_mask.data())), n_y, n_x,
                    thrust::raw_pointer_cast(device_labels.data()), thrust::raw_pointer_cast(device_count.data()),
                    thrust::raw_pointer_cast(device_scratch.data()), stream);

  std::vector<unsigned int> host_labels(n);
  unsigned int count = 0;
  gpuMemcpyDeviceToHostAsync(&count, thrust::raw_pointer_cast(device_count.data()), sizeof(unsigned int), stream);
  gpuMemcpyDeviceToHostAsync(thrust::raw_pointer_cast(host_labels.data()),
                             thrust::raw_pointer_cast(device_labels.data()), n * sizeof(unsigned int), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

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

TEST_CASE("gpu_label_islands rejects an empty plane", "[gpu][islands]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  DeviceVector<unsigned char> mask(1);
  DeviceVector<unsigned int> labels(1);
  DeviceVector<unsigned int> count(1);
  DeviceVector<unsigned int> scratch(1);

  REQUIRE_THROWS_AS(gpu_label_islands(reinterpret_cast<const bool *>(thrust::raw_pointer_cast(mask.data())), 0, 1,
                                      thrust::raw_pointer_cast(labels.data()), thrust::raw_pointer_cast(count.data()),
                                      thrust::raw_pointer_cast(scratch.data()), stream),
                    std::runtime_error);
  REQUIRE_THROWS_AS(gpu_label_islands(reinterpret_cast<const bool *>(thrust::raw_pointer_cast(mask.data())), 1, 0,
                                      thrust::raw_pointer_cast(labels.data()), thrust::raw_pointer_cast(count.data()),
                                      thrust::raw_pointer_cast(scratch.data()), stream),
                    std::runtime_error);

  gpuStreamDestroy(stream);
}
