#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gpu_band_workspace.cuh"
#include "device_vector.hpp"
#include "gpu_process_band.cuh"
#include "gpu_runtime.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::GpuBandInput;
using breifast::gpu::GpuBandResult;
using breifast::gpu::GpuBandInputSizes;
using breifast::gpu::GpuBandStatus;
using breifast::gpu::GpuBandWorkspace;
using breifast::gpu::GpuBandWorkspaceSizes;
using breifast::gpu::GpuStream;
using breifast::gpu::gpu_process_band;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;

namespace
{
struct Fixture
{
  static constexpr std::size_t n_y = 3;
  static constexpr std::size_t n_x = 4;
  static constexpr std::size_t n_pixels = n_y * n_x;
  static constexpr std::size_t n_channels = 2;
  static constexpr std::size_t histogram_bins = 10;

  DeviceVector<float> cube;
  DeviceVector<float> weights;
  DeviceVector<float> selfnoise_offset;
  DeviceVector<float> selfnoise_raw;
  DeviceVector<float> dde_noise;
  DeviceVector<uint8_t> accept_mask;
  DeviceVector<uint8_t> nonzero_channels;

  GpuBandWorkspace workspace;
  GpuBandResult result;
  GpuBandInput input;
  cudaStream_t stream{};

  static GpuBandWorkspaceSizes workspace_sizes()
  {
    return GpuBandWorkspaceSizes{
        .n_pixels = n_pixels,
        .stamp_capacity = 1,
        .stamp_pixels = 1,
        .beam_stamp_pixels = 0,
        .histogram_bins = histogram_bins,
    };
  }

  Fixture(bool nonzero_value, bool accept_value, bool use_rms)
      : workspace(workspace_sizes()), result(n_pixels, histogram_bins)
  {
    gpuStreamCreate(&stream);

    cube.assign(n_pixels * n_channels, 1.F);
    weights.assign(n_channels, 1.F);
    selfnoise_offset.assign(n_pixels, 0.F);
    selfnoise_raw.assign(n_pixels, 1.F);
    dde_noise.assign(n_pixels, 0.F);
    accept_mask.assign(n_pixels * n_channels, accept_value ? 1U : 0U);
    nonzero_channels.assign(n_channels, nonzero_value ? 1U : 0U);

    input = GpuBandInput{
        .sizes = GpuBandInputSizes{
            .cube = n_pixels * n_channels,
            .weights = n_channels,
            .nonzero_channels = n_channels,
            .accept_mask = n_pixels * n_channels,
            .selfnoise_offset = n_pixels,
            .selfnoise_raw = n_pixels,
            .dde_noise = n_pixels,
        },
        .cube = cube.data(),
        .weights = weights.data(),
        .nonzero_channels = reinterpret_cast<const bool *>(nonzero_channels.data()),
        .accept_mask = reinterpret_cast<const bool *>(accept_mask.data()),
        .selfnoise_offset = selfnoise_offset.data(),
        .selfnoise_raw = selfnoise_raw.data(),
        .dde_noise = dde_noise.data(),
        .n_y = n_y,
        .n_x = n_x,
        .n_pixels = n_pixels,
        .n_channels = n_channels,
        .band_start = 0,
        .band_stop = 2,
        .boxsize = 1,
        .inner_boxsize = 0,
        .beam_halfsize = 0,
        .histogram_detail = 0,
        .max_sigma_bin = 5,
        .selfnoise_scale_factor = 1.F,
        .inv_band_size = 0.5F,
        .inv_weights_sum = 0.5F,
        .ddenoise_scaling_factor = 1.F,
        .reject_noise_threshold = 0.F,
        .threshold = 2.F,
        .max_to_std_ratio_value = 1.F,
        .use_rms_noise_estimator = use_rms,
        .subtract_mean = false,
    };
  }

  ~Fixture()
  {
    gpuStreamSynchronize(stream);
    gpuStreamDestroy(stream);
  }
};

float copy_scalar(const float *value, GpuStream stream)
{
  float host_value = 0.F;
  gpuMemcpyDeviceToHostAsync(&host_value, value, sizeof(float), stream);
  gpuStreamSynchronize(stream);
  return host_value;
}

void set_accept_mask(Fixture &fixture, const std::vector<uint8_t> &mask)
{
  std::vector<uint8_t> host_mask(mask.begin(), mask.end());
  gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(fixture.accept_mask.data()),
                             thrust::raw_pointer_cast(host_mask.data()), mask.size(), fixture.stream);
  gpuStreamSynchronize(fixture.stream);
}
} // namespace

TEST_CASE("gpu_process_band skips a band with no nonzero channel", "[gpu][process_band]")
{
  Fixture fixture(false, true, true);
  CHECK(gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream) ==
        GpuBandStatus::skipped);
}

TEST_CASE("gpu_process_band skips a band whose accept mask is empty", "[gpu][process_band]")
{
  Fixture fixture(true, false, true);
  CHECK(gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream) ==
        GpuBandStatus::skipped);
}

TEST_CASE("gpu_process_band runs the RMS noise stage and reports no detections", "[gpu][process_band]")
{
  Fixture fixture(true, true, true);
  const auto status = gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream);
  CHECK(status == GpuBandStatus::no_detections);

  CHECK_THAT(copy_scalar(fixture.result.rms_floor, fixture.stream), WithinRel(1.F, 1e-6));
  CHECK_THAT(copy_scalar(fixture.result.median_noise, fixture.stream), WithinRel(1.F, 1e-6));
  CHECK_THAT(copy_scalar(fixture.result.global_mad, fixture.stream), WithinRel(1.4826F, 1e-6));

  std::vector<unsigned long long> histogram(Fixture::histogram_bins, 1ULL);
  gpuMemcpyDeviceToHostAsync(thrust::raw_pointer_cast(histogram.data()), fixture.result.histogram0,
                             Fixture::histogram_bins * sizeof(unsigned long long), fixture.stream);
  gpuStreamSynchronize(fixture.stream);
  for (const auto bin : histogram)
  {
    CHECK(bin == 0ULL);
  }
}

TEST_CASE("gpu_process_band runs the MAD noise stage and reports no detections", "[gpu][process_band]")
{
  Fixture fixture(true, true, false);
  const auto status = gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream);
  CHECK(status == GpuBandStatus::no_detections);

  CHECK_THAT(copy_scalar(fixture.result.global_mad, fixture.stream), WithinRel(1.4826F, 1e-6));
  CHECK_THAT(copy_scalar(fixture.result.median_noise, fixture.stream), WithinRel(1.4826F, 1e-6));
}

TEST_CASE("gpu_process_band rejects invalid input", "[gpu][process_band]")
{
  Fixture fixture(true, true, true);

  auto bad = fixture.input;
  bad.band_stop = 3;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream),
                    ContainsSubstring("invalid band range"));

  bad = fixture.input;
  bad.max_sigma_bin = 0;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream),
                    ContainsSubstring("max_sigma_bin"));

  bad = fixture.input;
  bad.cube = nullptr;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream),
                    ContainsSubstring("required input pointer"));

  bad = fixture.input;
  bad.sizes.cube = 1;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream),
                    ContainsSubstring("cube is too small"));

  bad = fixture.input;
  bad.sizes.accept_mask = 1;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream),
                    ContainsSubstring("accept_mask is too small"));

  bad = fixture.input;
  bad.n_y = std::numeric_limits<std::size_t>::max();
  bad.n_x = 2;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream),
                    ContainsSubstring("overflows"));
}

TEST_CASE("gpu_process_band errors when island count exceeds stamp_capacity", "[gpu][process_band]")
{
  Fixture fixture(true, true, true);
  fixture.input.threshold = 0.5F;

  std::vector<uint8_t> mask(Fixture::n_pixels * Fixture::n_channels, 0);
  mask[0] = 1;
  mask[Fixture::n_pixels - 1] = 1;
  set_accept_mask(fixture, mask);

  CHECK_THROWS_WITH(gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream),
                    ContainsSubstring("exceeds island capacity"));
}

TEST_CASE("gpu_process_band reaches the detection stage when capacity is sufficient", "[gpu][process_band]")
{
  Fixture fixture(true, true, true);
  fixture.input.threshold = 0.5F;

  std::vector<uint8_t> mask(Fixture::n_pixels * Fixture::n_channels, 0);
  mask[0] = 1;
  set_accept_mask(fixture, mask);

  CHECK(gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream) ==
        GpuBandStatus::detections);
}
