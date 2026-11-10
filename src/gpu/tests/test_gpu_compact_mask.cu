#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "device_vector.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_runtime.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_compact_mask;
using Catch::Matchers::Equals;

namespace
{
std::vector<size_t> run_compact(const std::vector<unsigned char> &mask)
{
  const size_t n = mask.size();

  std::vector<unsigned char> host_mask(mask.begin(), mask.end());
  DeviceVector<unsigned char> device_mask(n);
  DeviceVector<size_t> device_indices(n);
  DeviceVector<size_t> device_count(1);
  DeviceVector<unsigned int> device_scratch(n);

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_mask.data()), thrust::raw_pointer_cast(host_mask.data()),
                             n * sizeof(unsigned char), stream);
  gpu_compact_mask(reinterpret_cast<const bool *>(thrust::raw_pointer_cast(device_mask.data())), n,
                   thrust::raw_pointer_cast(device_indices.data()), thrust::raw_pointer_cast(device_count.data()),
                   thrust::raw_pointer_cast(device_scratch.data()), stream);

  size_t count = 0;
  std::vector<size_t> host_indices(n);
  gpuMemcpyDeviceToHostAsync(&count, thrust::raw_pointer_cast(device_count.data()), sizeof(size_t), stream);
  gpuMemcpyDeviceToHostAsync(thrust::raw_pointer_cast(host_indices.data()),
                             thrust::raw_pointer_cast(device_indices.data()), n * sizeof(size_t), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  return std::vector<size_t>(host_indices.begin(), host_indices.begin() + static_cast<long>(count));
}
} // namespace

TEST_CASE("gpu_compact_mask returns the true indices in raster order", "[gpu][compact_mask]")
{
  CHECK_THAT(run_compact({1, 0, 1, 1, 0, 0, 1, 0}), Equals(std::vector<size_t>{0, 2, 3, 6}));
  CHECK_THAT(run_compact({1, 0, 1, 0, 1, 0}), Equals(std::vector<size_t>{0, 2, 4}));
}

TEST_CASE("gpu_compact_mask handles all-false, all-true and single masks", "[gpu][compact_mask]")
{
  CHECK_THAT(run_compact(std::vector<unsigned char>(5, 0)), Equals(std::vector<size_t>{}));
  CHECK_THAT(run_compact(std::vector<unsigned char>(4, 1)), Equals(std::vector<size_t>{0, 1, 2, 3}));
  CHECK_THAT(run_compact({1, 0, 0, 0}), Equals(std::vector<size_t>{0}));
  CHECK_THAT(run_compact({0, 0, 0, 1}), Equals(std::vector<size_t>{3}));
}

TEST_CASE("gpu_compact_mask matches the true indices of a seeded pattern", "[gpu][compact_mask]")
{
  std::vector<unsigned char> mask(257);
  std::vector<size_t> expected;
  for (size_t i = 0; i < mask.size(); ++i)
  {
    mask[i] = static_cast<unsigned char>((i * 7 + 3) % 5 < 2);
    if (mask[i] != 0)
    {
      expected.push_back(i);
    }
  }

  CHECK_THAT(run_compact(mask), Equals(expected));
}

TEST_CASE("gpu_compact_mask fills a multi-block mask against a host reference", "[gpu][compact_mask]")
{
  std::vector<unsigned char> mask(10000);
  std::vector<size_t> expected;
  for (size_t i = 0; i < mask.size(); ++i)
  {
    mask[i] = static_cast<unsigned char>((i % 3 == 0) || (i % 7 == 1));
    if (mask[i] != 0)
    {
      expected.push_back(i);
    }
  }

  CHECK_THAT(run_compact(mask), Equals(expected));
}

TEST_CASE("gpu_compact_mask rejects an empty mask", "[gpu][compact_mask]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  DeviceVector<size_t> indices(1);
  DeviceVector<size_t> count(1);
  DeviceVector<unsigned int> scratch(1);

  REQUIRE_THROWS_AS(gpu_compact_mask(nullptr, 0, thrust::raw_pointer_cast(indices.data()),
                                     thrust::raw_pointer_cast(count.data()), thrust::raw_pointer_cast(scratch.data()),
                                     stream),
                    std::runtime_error);

  gpuStreamDestroy(stream);
}
