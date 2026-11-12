#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_band.hpp"
#include "gpu_test_utils.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_process_band;
using breifast::gpu::GpuBand;
using breifast::gpu::GpuBandResult;
using breifast::gpu::GpuBandWorkspace;
using breifast::gpu::GpuPlaneInput;
using breifast::gpu::Stream;
using breifast::gpu::test::download;
using breifast::gpu::test::download_scalar;
using breifast::gpu::test::upload;
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

  GpuBandWorkspace workspace;
  GpuBandResult result;
  GpuPlaneInput plane;
  GpuBand band{.start = 0, .stop = 2};
  // Declared after the buffers: destroyed first, it waits for their work.
  Stream stream;

  Fixture(bool accept_value, bool use_rms)
  {
    cube = upload(std::vector<float>(n_pixels * n_channels, 1.F));
    weights = upload(std::vector<float>(n_channels, 1.F));
    selfnoise_offset = upload(std::vector<float>(n_pixels, 0.F));
    selfnoise_raw = upload(std::vector<float>(n_pixels, 1.F));
    dde_noise = upload(std::vector<float>(n_pixels, 0.F));
    accept_mask = upload(std::vector<bool>(n_pixels * n_channels, accept_value));

    plane = GpuPlaneInput{
        .cube = cube.data(),
        .weights = weights.data(),
        .accept_mask = accept_mask.data(),
        .selfnoise_offset = selfnoise_offset.data(),
        .selfnoise_raw = selfnoise_raw.data(),
        .dde_noise = dde_noise.data(),
        .n_y = n_y,
        .n_x = n_x,
        .n_channels = n_channels,
        .boxsize = 1,
        .inner_boxsize = 0,
        .histogram_detail = 0,
        .max_sigma_bin = 5,
        .ddenoise_scaling_factor = 1.F,
        .reject_noise_threshold = 0.F,
        .threshold = 2.F,
        .max_to_std_ratio = 1.F,
        .use_rms_noise_estimator = use_rms,
    };
  }
};

// Replaces the accept mask, and points the plane input at the new buffer.
void set_accept_mask(Fixture &fixture, const std::vector<bool> &mask)
{
  fixture.accept_mask = upload(mask);
  fixture.plane.accept_mask = fixture.accept_mask.data();
}
} // namespace

TEST_CASE("gpu_process_band skips a band whose accept mask is empty", "[gpu][process_band]")
{
  Fixture fixture(false, true);
  CHECK_FALSE(gpu_process_band(fixture.plane, fixture.band, fixture.workspace, fixture.result, fixture.stream));
}

TEST_CASE("gpu_process_band runs the RMS noise stage and reports no detections", "[gpu][process_band]")
{
  Fixture fixture(true, true);
  CHECK(gpu_process_band(fixture.plane, fixture.band, fixture.workspace, fixture.result, fixture.stream));
  CHECK(fixture.result.label_count == 0U);

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
  Fixture fixture(true, false);
  CHECK(gpu_process_band(fixture.plane, fixture.band, fixture.workspace, fixture.result, fixture.stream));
  CHECK(fixture.result.label_count == 0U);

  CHECK_THAT(download_scalar(fixture.result.global_mad.data(), fixture.stream), WithinRel(1.4826F, 1e-6));
  CHECK_THAT(download_scalar(fixture.result.median_noise.data(), fixture.stream), WithinRel(1.4826F, 1e-6));
}

TEST_CASE("gpu_process_band reaches the detection stage", "[gpu][process_band]")
{
  Fixture fixture(true, true);
  fixture.plane.threshold = 0.5F;

  std::vector<bool> mask(Fixture::n_pixels * Fixture::n_channels, false);
  mask[0] = true;
  set_accept_mask(fixture, mask);

  CHECK(gpu_process_band(fixture.plane, fixture.band, fixture.workspace, fixture.result, fixture.stream));
  CHECK(fixture.result.label_count == 1U);
}
