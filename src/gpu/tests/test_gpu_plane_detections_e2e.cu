#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "gpu_band_result.cuh"
#include "gpu_test_utils.hpp"
#include "gpu_plane_detections.hpp"
#include "gpu_process_band.cuh"

using breifast::DetectionArrays;
using breifast::gpu::assemble_plane_bands;
using breifast::gpu::assemble_plane_detections;
using breifast::gpu::copy_band_jobs;
using breifast::gpu::DeviceVector;
using breifast::gpu::GpuBandHostResult;
using breifast::gpu::GpuBandInput;
using breifast::gpu::GpuBandInputSizes;
using breifast::gpu::GpuBandWorkspaceSizes;
using breifast::gpu::GpuBandJob;
using breifast::gpu::GpuBandLabels;
using breifast::gpu::GpuBandStatus;
using breifast::gpu::gpu_process_plane;
using breifast::gpu::GpuPlaneHostResult;
using breifast::gpu::test::upload;
using Catch::Approx;

namespace
{
struct DetectionFixture
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

  std::optional<GpuBandJob> job;

  DetectionFixture()
  {
    cube = upload(std::vector<float>(n_pixels * n_channels, 1.F));
    weights = upload(std::vector<float>(n_channels, 1.F));
    selfnoise_offset = upload(std::vector<float>(n_pixels, 0.F));
    selfnoise_raw = upload(std::vector<float>(n_pixels, 1.F));
    dde_noise = upload(std::vector<float>(n_pixels, 0.F));
    std::vector<bool> host_accept_mask(n_pixels * n_channels, false);
    host_accept_mask[0] = true;
    accept_mask = upload(host_accept_mask);
    nonzero_channels = upload(std::vector<bool>(n_channels, true));

    const GpuBandInput input{
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
        .histogram_detail = 1,
        .max_sigma_bin = 5,
        .selfnoise_scale_factor = 1.F,
        .inv_band_size = 0.5F,
        .ddenoise_scaling_factor = 1.F,
        .reject_noise_threshold = 0.F,
        .threshold = 0.5F,
        .max_to_std_ratio_value = 1.F,
        .use_rms_noise_estimator = true,
        .subtract_mean = false,
    };
    const GpuBandWorkspaceSizes workspace_sizes{
        .n_pixels = n_pixels,
        .stamp_capacity = 1,
        .stamp_pixels = 1,
        .beam_stamp_pixels = 0,
        .histogram_bins = histogram_bins,
    };
    job.emplace(input, workspace_sizes, n_pixels, histogram_bins);
  }
};
} // namespace

TEST_CASE("GPU plane job produces an assembled detection row", "[gpu][plane_detections][e2e]")
{
  DetectionFixture fixture;
  std::vector<GpuBandJob> jobs;
  jobs.push_back(std::move(*fixture.job));

  GpuPlaneHostResult plane;
  gpu_process_plane(jobs, plane);

  REQUIRE(plane.statuses.size() == 1);
  CHECK(plane.statuses[0] == GpuBandStatus::detections);
  REQUIRE(plane.bands.size() == 1);
  CHECK(plane.bands[0].label_count == 1U);
  CHECK(plane.histograms[2][6] == 1ULL);

  const std::vector<GpuBandLabels> labels = {{"I", "band"}};
  DetectionArrays detections;
  assemble_plane_detections(plane, labels, 1.F, true, detections);

  REQUIRE(detections.size() == 1);
  CHECK(detections.stokes_label()[0] == "I");
  CHECK(detections.band_label()[0] == "band");
  CHECK(detections.peak_y()[0] == 0);
  CHECK(detections.peak_x()[0] == 0);
  CHECK(detections.island_size()[0] == 1);
  CHECK(detections.flux()[0] == Approx(1.F));
  CHECK(detections.local_noise()[0] == Approx(1.F));
  CHECK(detections.local_noise_median_ratio()[0] == Approx(1.F));
  CHECK(detections.signal_to_noise()[0] == Approx(1.F));
  CHECK(detections.signal_to_noise_rms()[0] == Approx(0.F));
  CHECK(detections.signal_to_noise_extended_box()[0] == Approx(0.F));
  CHECK(detections.signal_to_noise_local_mad()[0] == Approx(0.F));
  CHECK(detections.signal_to_noise_global_mad()[0] == Approx(1.F / 1.4826F));
  CHECK(detections.self_noise()[0] == Approx(1.F));
  CHECK(detections.excess_noise_factor()[0] == Approx(1.F));
  CHECK(detections.beam_chi2()[0] == Approx(0.F));
  CHECK(detections.boost()[0] == Approx(0.F));

}
