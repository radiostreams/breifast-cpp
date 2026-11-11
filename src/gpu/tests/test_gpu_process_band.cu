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
#include "gpu_process_band.cuh"
#include "gpu_test_utils.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::GpuBandInput;
using breifast::gpu::GpuBandResult;
using breifast::gpu::GpuBandInputSizes;
using breifast::gpu::GpuBandStatus;
using breifast::gpu::GpuBandWorkspace;
using breifast::gpu::GpuBandWorkspaceSizes;
using breifast::gpu::gpu_process_band;
using breifast::gpu::Stream;
using breifast::gpu::test::download;
using breifast::gpu::test::download_scalar;
using breifast::gpu::test::upload;
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
  DeviceVector<bool> accept_mask;
  DeviceVector<bool> nonzero_channels;

  GpuBandWorkspace workspace;
  GpuBandResult result;
  GpuBandInput input;
  // Declared after the buffers: destroyed first, it waits for their work.
  Stream stream;

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
    cube = upload(std::vector<float>(n_pixels * n_channels, 1.F));
    weights = upload(std::vector<float>(n_channels, 1.F));
    selfnoise_offset = upload(std::vector<float>(n_pixels, 0.F));
    selfnoise_raw = upload(std::vector<float>(n_pixels, 1.F));
    dde_noise = upload(std::vector<float>(n_pixels, 0.F));
    accept_mask = upload(std::vector<bool>(n_pixels * n_channels, accept_value));
    nonzero_channels = upload(std::vector<bool>(n_channels, nonzero_value));

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
        .nonzero_channels = nonzero_channels.data(),
        .accept_mask = accept_mask.data(),
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
        .ddenoise_scaling_factor = 1.F,
        .reject_noise_threshold = 0.F,
        .threshold = 2.F,
        .max_to_std_ratio_value = 1.F,
        .use_rms_noise_estimator = use_rms,
        .subtract_mean = false,
    };
  }
};

// Replaces the accept mask, and points the band input at the new buffer.
void set_accept_mask(Fixture &fixture, const std::vector<bool> &mask)
{
  fixture.accept_mask = upload(mask);
  fixture.input.accept_mask = fixture.accept_mask.data();
}
} // namespace

TEST_CASE("gpu_process_band skips a band with no nonzero channel", "[gpu][process_band]")
{
  Fixture fixture(false, true, true);
  CHECK(gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream.get()) ==
        GpuBandStatus::skipped);
}

TEST_CASE("gpu_process_band skips a band whose accept mask is empty", "[gpu][process_band]")
{
  Fixture fixture(true, false, true);
  CHECK(gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream.get()) ==
        GpuBandStatus::skipped);
}

TEST_CASE("gpu_process_band runs the RMS noise stage and reports no detections", "[gpu][process_band]")
{
  Fixture fixture(true, true, true);
  const auto status = gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream.get());
  CHECK(status == GpuBandStatus::no_detections);

  CHECK_THAT(download_scalar(fixture.result.rms_floor.data(), fixture.stream), WithinRel(1.F, 1e-6));
  CHECK_THAT(download_scalar(fixture.result.median_noise.data(), fixture.stream), WithinRel(1.F, 1e-6));
  CHECK_THAT(download_scalar(fixture.result.global_mad.data(), fixture.stream), WithinRel(1.4826F, 1e-6));

  const auto histogram = download(fixture.result.histogram0, fixture.stream);
  REQUIRE(histogram.size() == Fixture::histogram_bins);
  for (const auto bin : histogram)
  {
    CHECK(bin == 0ULL);
  }
}

TEST_CASE("gpu_process_band runs the MAD noise stage and reports no detections", "[gpu][process_band]")
{
  Fixture fixture(true, true, false);
  const auto status = gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream.get());
  CHECK(status == GpuBandStatus::no_detections);

  CHECK_THAT(download_scalar(fixture.result.global_mad.data(), fixture.stream), WithinRel(1.4826F, 1e-6));
  CHECK_THAT(download_scalar(fixture.result.median_noise.data(), fixture.stream), WithinRel(1.4826F, 1e-6));
}

TEST_CASE("gpu_process_band rejects invalid input", "[gpu][process_band]")
{
  Fixture fixture(true, true, true);

  auto bad = fixture.input;
  bad.band_stop = 3;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream.get()),
                    ContainsSubstring("invalid band range"));

  bad = fixture.input;
  bad.max_sigma_bin = 0;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream.get()),
                    ContainsSubstring("max_sigma_bin"));

  bad = fixture.input;
  bad.cube = nullptr;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream.get()),
                    ContainsSubstring("required input pointer"));

  bad = fixture.input;
  bad.sizes.cube = 1;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream.get()),
                    ContainsSubstring("cube is too small"));

  bad = fixture.input;
  bad.sizes.accept_mask = 1;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream.get()),
                    ContainsSubstring("accept_mask is too small"));

  bad = fixture.input;
  bad.n_y = std::numeric_limits<std::size_t>::max();
  bad.n_x = 2;
  CHECK_THROWS_WITH(gpu_process_band(bad, fixture.workspace, fixture.result, fixture.stream.get()),
                    ContainsSubstring("overflows"));
}

TEST_CASE("gpu_process_band errors when island count exceeds stamp_capacity", "[gpu][process_band]")
{
  Fixture fixture(true, true, true);
  fixture.input.threshold = 0.5F;

  std::vector<bool> mask(Fixture::n_pixels * Fixture::n_channels, false);
  mask[0] = true;
  mask[Fixture::n_pixels - 1] = true;
  set_accept_mask(fixture, mask);

  CHECK_THROWS_WITH(gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream.get()),
                    ContainsSubstring("exceeds island capacity"));
}

TEST_CASE("gpu_process_band reaches the detection stage when capacity is sufficient", "[gpu][process_band]")
{
  Fixture fixture(true, true, true);
  fixture.input.threshold = 0.5F;

  std::vector<bool> mask(Fixture::n_pixels * Fixture::n_channels, false);
  mask[0] = true;
  set_accept_mask(fixture, mask);

  CHECK(gpu_process_band(fixture.input, fixture.workspace, fixture.result, fixture.stream.get()) ==
        GpuBandStatus::detections);
}
