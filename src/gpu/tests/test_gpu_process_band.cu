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
#include "gpu_buffers.hpp"
#include "gpu_process_band.cuh"
#include "gpu_runtime.hpp"

using breifast::gpu::create_band_result;
using breifast::gpu::create_band_workspace;
using breifast::gpu::destroy_band_result;
using breifast::gpu::destroy_band_workspace;
using breifast::gpu::DeviceVector;
using breifast::gpu::GpuBandInput;
using breifast::gpu::GpuBandResult;
using breifast::gpu::GpuBandStatus;
using breifast::gpu::GpuBandWorkspace;
using breifast::gpu::GpuBandWorkspaceSizes;
using breifast::gpu::GpuStream;
using breifast::gpu::gpu_process_band;
using breifast::gpu::PinnedVector;
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
  DeviceVector<float> dde_noise;
  DeviceVector<uint8_t> accept_mask;
  DeviceVector<uint8_t> nonzero_channels;

  GpuBandWorkspace workspace;
  GpuBandResult result;
  GpuBandInput input;
  cudaStream_t stream{};

  Fixture(bool nonzero_value, bool accept_value, bool use_rms)
  {
    gpuStreamCreate(&stream);

    cube.assign(n_pixels * n_channels, 1.F);
    weights.assign(n_channels, 1.F);
    selfnoise_offset.assign(n_pixels, 0.F);
    dde_noise.assign(n_pixels, 0.F);
    accept_mask.assign(n_pixels * n_channels, accept_value ? 1U : 0U);
    nonzero_channels.assign(n_channels, nonzero_value ? 1U : 0U);

    GpuBandWorkspaceSizes sizes;
    sizes.n_pixels = n_pixels;
    sizes.stamp_capacity = 1;
    sizes.stamp_pixels = 1;
    sizes.beam_stamp_pixels = 0;
    sizes.histogram_bins = histogram_bins;
    workspace = create_band_workspace(sizes, stream);
    result = create_band_result(n_pixels, histogram_bins, stream);

    input.sizes.cube = n_pixels * n_channels;
    input.sizes.weights = n_channels;
    input.sizes.nonzero_channels = n_channels;
    input.sizes.accept_mask = n_pixels * n_channels;
    input.sizes.selfnoise_offset = n_pixels;
    input.sizes.dde_noise = n_pixels;

    input.cube = thrust::raw_pointer_cast(cube.data());
    input.weights = thrust::raw_pointer_cast(weights.data());
    input.nonzero_channels = reinterpret_cast<const bool *>(thrust::raw_pointer_cast(nonzero_channels.data()));
    input.accept_mask = reinterpret_cast<const bool *>(thrust::raw_pointer_cast(accept_mask.data()));
    input.selfnoise_offset = thrust::raw_pointer_cast(selfnoise_offset.data());
    input.dde_noise = thrust::raw_pointer_cast(dde_noise.data());

    input.n_y = n_y;
    input.n_x = n_x;
    input.n_pixels = n_pixels;
    input.n_channels = n_channels;
    input.band_start = 0;
    input.band_stop = 2;
    input.boxsize = 1;
    input.inner_boxsize = 0;
    input.beam_halfsize = 0;
    input.histogram_detail = 0;
    input.max_sigma_bin = 5;
    input.selfnoise_scale_factor = 1.F;
    input.inv_band_size = 0.5F;
    input.inv_weights_sum = 0.5F;
    input.ddenoise_scaling_factor = 1.F;
    input.reject_noise_threshold = 0.F;
    input.threshold = 2.F;
    input.max_to_std_ratio_value = 1.F;
    input.use_rms_noise_estimator = use_rms;
    input.subtract_mean = false;
  }

  ~Fixture()
  {
    destroy_band_result(result, stream);
    destroy_band_workspace(workspace, stream);
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

  PinnedVector<unsigned long long> histogram(Fixture::histogram_bins, 1ULL);
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
