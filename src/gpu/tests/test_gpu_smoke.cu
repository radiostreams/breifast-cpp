#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "gpu_test_utils.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::Stream;
using breifast::gpu::test::download;
using breifast::gpu::test::upload;

TEST_CASE("device vectors round trip a float array", "[gpu][smoke]")
{
  // Odd size, so nothing can assume a multiple of the block or vector width.
  constexpr size_t n = 1'000'003;
  std::vector<float> host(n);
  for (size_t i = 0; i < n; ++i)
  {
    host[i] = 0.5F * static_cast<float>(i);
  }

  auto device = upload(host);
  REQUIRE(device.size() == n);

  // The allocation comes from device memory.
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, device.data()) == cudaSuccess);
  REQUIRE(attributes.type == cudaMemoryTypeDevice);

  const Stream stream;
  const bool round_trip = download(device, stream) == host; // not decomposed: Catch would print a million floats
  CHECK(round_trip);

  // A moved-to vector owns the moved-from one's allocation.
  const float *const allocation = device.data();
  const DeviceVector<float> moved = std::move(device);
  CHECK(moved.data() == allocation);
  CHECK(moved.size() == n);
}

TEST_CASE("CHECK_GPU throws a GpuError that carries the CUDA error", "[gpu][smoke]")
{
  try
  {
    CHECK_GPU(cudaSetDevice(-1));
    FAIL("CHECK_GPU did not throw");
  }
  catch (const breifast::gpu::GpuError &error)
  {
    CHECK(error.error() == cudaErrorInvalidDevice);
    CHECK_THAT(error.what(), Catch::Matchers::ContainsSubstring("cudaSetDevice(-1)"));
    CHECK_THAT(error.what(), !Catch::Matchers::ContainsSubstring("restart"));
  }
  // A failed call is also recorded as the last error: clear it, or the next launch check would report it.
  (void)cudaGetLastError();
}
