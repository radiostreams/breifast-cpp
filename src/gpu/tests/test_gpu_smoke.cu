#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <thrust/copy.h>

#include <cstddef>
#include <cstring>
#include <utility>

#include "gpu_buffers.hpp"
#include "gpu_runtime.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::PinnedVector;

TEST_CASE("pinned and device vectors round trip a float array", "[gpu][smoke]")
{
  // Odd size, so nothing can assume a multiple of the block or vector width.
  constexpr size_t n = 1'000'003;
  constexpr size_t bytes = n * sizeof(float);
  PinnedVector<float> host_in(n);
  PinnedVector<float> host_out(n);
  DeviceVector<float> device(n);

  REQUIRE(host_in.size() == n);
  REQUIRE(host_out.size() == n);
  REQUIRE(device.size() == n);

  float *const in = thrust::raw_pointer_cast(host_in.data());
  float *const out = thrust::raw_pointer_cast(host_out.data());
  float *const device_ptr = thrust::raw_pointer_cast(device.data());

  // The allocations come from the right memory: page-locked host memory and device memory.
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, in) == cudaSuccess);
  REQUIRE(attributes.type == cudaMemoryTypeHost);
  REQUIRE(cudaPointerGetAttributes(&attributes, device_ptr) == cudaSuccess);
  REQUIRE(attributes.type == cudaMemoryTypeDevice);

  for (size_t i = 0; i < n; ++i)
  {
    in[i] = 0.5F * static_cast<float>(i);
    out[i] = -1.F;
  }

  // Round trip through the runtime header on an explicit stream.
  cudaStream_t stream{};
  gpuStreamCreate(&stream);
  gpuMemcpyHostToDeviceAsync(device_ptr, in, bytes, stream);
  gpuMemcpyDeviceToHostAsync(out, device_ptr, bytes, stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);
  REQUIRE(std::memcmp(in, out, bytes) == 0);

  // Round trip through Thrust copies.
  for (size_t i = 0; i < n; ++i)
  {
    out[i] = -1.F;
  }
  thrust::copy(host_in.begin(), host_in.end(), device.begin());
  thrust::copy(device.begin(), device.end(), host_out.begin());
  REQUIRE(std::memcmp(in, out, bytes) == 0);

  // A moved-to vector owns the data of the moved-from one.
  PinnedVector<float> moved = std::move(host_out);
  REQUIRE(moved.size() == n);
  REQUIRE(std::memcmp(in, thrust::raw_pointer_cast(moved.data()), bytes) == 0);
}
