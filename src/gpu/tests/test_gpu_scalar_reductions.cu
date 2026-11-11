#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "device_vector.hpp"
#include "gpu_kernels.hpp"
#include "gpu_runtime.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_any_true;

namespace
{
// A mask is a byte per value (0 or 1), reinterpreted as bool in the wrapper.
int run_any_true(const std::vector<unsigned char> &values)
{
  const size_t n = values.size();
  const size_t n_alloc = n == 0 ? 1 : n;

  std::vector<unsigned char> host_values(values.begin(), values.end());
  DeviceVector<unsigned char> device_values(n_alloc);
  DeviceVector<int> device_out(1);

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  if (n > 0)
  {
    gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_values.data()), thrust::raw_pointer_cast(host_values.data()),
                               n * sizeof(unsigned char), stream);
  }

  const bool *mask = reinterpret_cast<const bool *>(thrust::raw_pointer_cast(device_values.data()));
  gpu_any_true(mask, n, thrust::raw_pointer_cast(device_out.data()), stream);

  int result = -1;
  gpuMemcpyDeviceToHostAsync(&result, thrust::raw_pointer_cast(device_out.data()), sizeof(int), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);
  return result;
}
} // namespace

TEST_CASE("gpu_any_true returns 0 for all-false and nonzero for all-true across sizes", "[gpu][scalar_reductions]")
{
  // Covers the empty-block tail (sizes just under/over 256) and a multi-block grid.
  for (const size_t n : {1UL, 2UL, 31UL, 32UL, 33UL, 63UL, 64UL, 65UL, 255UL, 256UL, 257UL, 511UL, 512UL, 513UL,
                         1000UL, 4095UL, 4096UL, 4097UL, 10000UL})
  {
    INFO("n = " << n);
    CHECK(run_any_true(std::vector<unsigned char>(n, 0)) == 0);
    CHECK(run_any_true(std::vector<unsigned char>(n, 1)) != 0);
  }
}

TEST_CASE("gpu_any_true finds a single true at every position", "[gpu][scalar_reductions]")
{
  constexpr size_t n = 1000; // nearly four blocks, so the last block is partial
  for (size_t i = 0; i < n; ++i)
  {
    std::vector<unsigned char> mask(n, 0);
    mask[i] = 1;
    INFO("true at index " << i);
    CHECK(run_any_true(mask) != 0);
  }
}

TEST_CASE("gpu_any_true distinguishes an all-false mask from any single true", "[gpu][scalar_reductions]")
{
  // The same size as the position sweep, all false, so a mask that misses the true is caught.
  constexpr size_t n = 1000;
  CHECK(run_any_true(std::vector<unsigned char>(n, 0)) == 0);

  // A true only at the very last index of a partial tail block.
  std::vector<unsigned char> tail(n, 0);
  tail[n - 1] = 1;
  CHECK(run_any_true(tail) != 0);

  // A non-bool-representable byte still counts as true.
  std::vector<unsigned char> wide(n, 0);
  wide[n / 2] = 255;
  CHECK(run_any_true(wide) != 0);
}

TEST_CASE("gpu_any_true rejects an empty array", "[gpu][scalar_reductions]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  DeviceVector<int> out(1);
  int *out_ptr = thrust::raw_pointer_cast(out.data());

  REQUIRE_THROWS_AS(gpu_any_true(nullptr, 0, out_ptr, stream), std::runtime_error);

  gpuStreamDestroy(stream);
}
