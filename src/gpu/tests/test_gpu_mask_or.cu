#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "device_vector.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_runtime.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_mask_or;
using Catch::Matchers::Equals;

namespace
{
struct Result
{
  bool any;
  std::string mask; // one '0' or '1' per pixel
};

// Runs gpu_mask_or on channel-major masks given as one string of '0' and '1' per channel. `out_any` is pre-filled with
// a nonzero sentinel, so the call's own zeroing is exercised too.
Result run_gpu_mask_or(const std::vector<std::string> &channels, int band_start, int band_stop)
{
  const size_t n_pixels = channels.front().size();
  const size_t n_channels = channels.size();

  std::vector<uint8_t> host_mask(n_channels * n_pixels);
  for (size_t i_channel = 0; i_channel < n_channels; ++i_channel)
  {
    for (size_t i_pixel = 0; i_pixel < n_pixels; ++i_pixel)
    {
      host_mask[i_channel * n_pixels + i_pixel] = channels[i_channel][i_pixel] == '1' ? uint8_t{1} : uint8_t{0};
    }
  }

  std::vector<uint8_t> host_out(n_pixels);
  std::vector<int> host_any(1);

  DeviceVector<uint8_t> device_mask(n_channels * n_pixels);
  DeviceVector<uint8_t> device_out(n_pixels);
  DeviceVector<int> device_any(1);

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_mask.data()),
                             thrust::raw_pointer_cast(host_mask.data()), n_channels * n_pixels, stream);
  gpuMemsetAsync(thrust::raw_pointer_cast(device_any.data()), 0xFF, sizeof(int), stream);

  gpu_mask_or(reinterpret_cast<const bool *>(thrust::raw_pointer_cast(device_mask.data())), n_pixels, band_start,
              band_stop, reinterpret_cast<bool *>(thrust::raw_pointer_cast(device_out.data())),
              thrust::raw_pointer_cast(device_any.data()), stream);

  gpuMemcpyDeviceToHostAsync(thrust::raw_pointer_cast(host_out.data()),
                             thrust::raw_pointer_cast(device_out.data()), n_pixels, stream);
  gpuMemcpyDeviceToHostAsync(thrust::raw_pointer_cast(host_any.data()),
                             thrust::raw_pointer_cast(device_any.data()), sizeof(int), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  std::string mask;
  mask.reserve(n_pixels);
  for (size_t i_pixel = 0; i_pixel < n_pixels; ++i_pixel)
  {
    mask += host_out[i_pixel] ? '1' : '0';
  }
  return {host_any[0] != 0, mask};
}
} // namespace

TEST_CASE("gpu_mask_or matches a fixed 4x37 mask with a band that skips channel 0", "[gpu][mask_or]")
{
  // Same mask and expected output as the CPU or_accept_masks case (numpy seed 42, p(true) = 0.15):
  // np.any(mask[1:4], axis=0). 37 pixels is not a multiple of the 256-thread block.
  const std::vector<std::string> channels = {
      "0000100010000000010000000001000000001",
      "0000000000001010000001100000000100000",
      "1000000000110000000000011000000000100",
      "0000000000010100110000001000100000000",
  };
  const std::string expected = "1000000000111110110001111000100100100";

  const Result result = run_gpu_mask_or(channels, 1, 4);

  CHECK(result.any);
  CHECK_THAT(result.mask, Equals(expected));
}

TEST_CASE("gpu_mask_or ignores channels outside the band and zeroes a stale out_any", "[gpu][mask_or]")
{
  // The channels around the band [1, 3) are all true, and out_any starts nonzero (the harness pre-fills it with a
  // sentinel). Neither may leak into the result.
  const std::vector<std::string> channels = {
      "1111111111111111111",
      "0000000000000000000",
      "0000000000000000000",
      "1111111111111111111",
  };

  const Result result = run_gpu_mask_or(channels, 1, 3);

  CHECK(result.any == false);
  CHECK_THAT(result.mask, Equals("0000000000000000000"));
}

TEST_CASE("gpu_mask_or handles a single-channel band", "[gpu][mask_or]")
{
  // band [2, 3) is one channel: the result is exactly channel 2, and the true pixels of the other channels are ignored.
  const std::vector<std::string> channels = {
      "1111111111111111",
      "0000111100001111",
      "0010000000000100",
      "1111000000000000",
  };

  const Result result = run_gpu_mask_or(channels, 2, 3);

  CHECK(result.any);
  CHECK_THAT(result.mask, Equals("0010000000000100"));
}

TEST_CASE("gpu_mask_or covers the partial last block of a 1000-pixel plane", "[gpu][mask_or]")
{
  // 1000 pixels is 3 full 256-thread blocks plus a tail of 232. The only true pixels are in the last block, one of
  // them the very last pixel of the channel.
  constexpr size_t n_pixels = 1000;
  std::string channel0(n_pixels, '0');
  std::string channel1(n_pixels, '0');
  std::string channel2(n_pixels, '0');
  channel0[5] = '1';
  channel1[256] = '1';
  channel2[n_pixels - 1] = '1';

  std::string expected(n_pixels, '0');
  expected[5] = '1';
  expected[256] = '1';
  expected[n_pixels - 1] = '1';

  const Result result = run_gpu_mask_or({channel0, channel1, channel2}, 0, 3);

  CHECK(result.any);
  CHECK_THAT(result.mask, Equals(expected));
}

TEST_CASE("gpu_mask_or covers a plane that is an exact multiple of the block size", "[gpu][mask_or]")
{
  // 512 pixels is exactly 2 blocks, so there is no partial tail block. The true pixels sit in different channels.
  constexpr size_t n_pixels = 512;
  std::string channel0(n_pixels, '0');
  std::string channel1(n_pixels, '0');
  channel0[n_pixels - 1] = '1';
  channel1[256] = '1';

  std::string expected(n_pixels, '0');
  expected[256] = '1';
  expected[n_pixels - 1] = '1';

  const Result result = run_gpu_mask_or({channel0, channel1}, 0, 2);

  CHECK(result.any);
  CHECK_THAT(result.mask, Equals(expected));
}

TEST_CASE("gpu_mask_or covers a plane spanning many blocks", "[gpu][mask_or]")
{
  // 70001 pixels is 273 blocks plus a tail, so the atomic OR into out_any is fed by many blocks.
  constexpr size_t n_pixels = 70001;
  std::string channel0(n_pixels, '0');
  std::string channel1(n_pixels, '0');
  channel0[n_pixels - 1] = '1';
  channel1[0] = '1';

  std::string expected(n_pixels, '0');
  expected[0] = '1';
  expected[n_pixels - 1] = '1';

  const Result result = run_gpu_mask_or({channel0, channel1}, 0, 2);

  CHECK(result.any);
  CHECK_THAT(result.mask, Equals(expected));
}

TEST_CASE("gpu_mask_or rejects an invalid band or an empty plane", "[gpu][mask_or]")
{
  using Catch::Matchers::ContainsSubstring;
  CHECK_THROWS_WITH(gpu_mask_or(nullptr, 4, -1, 1, nullptr, nullptr), ContainsSubstring("invalid band"));
  CHECK_THROWS_WITH(gpu_mask_or(nullptr, 4, 1, 1, nullptr, nullptr), ContainsSubstring("invalid band"));
  CHECK_THROWS_WITH(gpu_mask_or(nullptr, 4, 2, 1, nullptr, nullptr), ContainsSubstring("invalid band"));
  CHECK_THROWS_WITH(gpu_mask_or(nullptr, 0, 0, 1, nullptr, nullptr), ContainsSubstring("zero pixels"));
}
