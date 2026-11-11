#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "gpu_kernels.hpp"
#include "gpu_test_utils.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_any_true;
using breifast::gpu::Stream;
using breifast::gpu::test::download_scalar;
using breifast::gpu::test::upload;

namespace
{
// A mask is a byte per value, reinterpreted as bool in the wrapper, so a byte other than 0 or 1 can be tested too.
int run_any_true(const std::vector<unsigned char> &values)
{
  const auto device_values = upload(values);
  DeviceVector<int> device_out(1);

  const Stream stream;
  gpu_any_true(reinterpret_cast<const bool *>(device_values.data()), values.size(), device_out, stream.get());
  return download_scalar(device_out.data(), stream);
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
