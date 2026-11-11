#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "gpu_band_result.cuh"
#include "gpu_test_utils.hpp"
#include "gpu_process_band.cuh"
#include "gpu_runtime.hpp"

using breifast::gpu::copy_band_jobs;
using breifast::gpu::DeviceVector;
using breifast::gpu::GpuBandInput;
using breifast::gpu::GpuBandInputSizes;
using breifast::gpu::GpuBandWorkspaceSizes;
using breifast::gpu::GpuBandJob;
using breifast::gpu::GpuBandHostResult;
using breifast::gpu::GpuBandStatus;
using breifast::gpu::gpu_process_bands;
using breifast::gpu::gpu_process_plane;
using breifast::gpu::GpuPlaneHostResult;
using breifast::gpu::test::upload;
using Catch::Matchers::WithinRel;

namespace
{
struct JobFixture
{
  static constexpr std::size_t n_y = 2;
  static constexpr std::size_t n_x = 2;
  static constexpr std::size_t n_pixels = n_y * n_x;
  static constexpr std::size_t n_channels = 1;
  static constexpr std::size_t histogram_bins = 4;

  DeviceVector<float> cube;
  DeviceVector<float> weights;
  DeviceVector<float> selfnoise_offset;
  DeviceVector<float> selfnoise_raw;
  DeviceVector<float> dde_noise;
  DeviceVector<bool> accept_mask;
  DeviceVector<bool> nonzero_channels;

  std::optional<GpuBandJob> job;

  JobFixture()
  {
    cube = upload(std::vector<float>(n_pixels * n_channels, 1.F));
    weights = upload(std::vector<float>(n_channels, 1.F));
    selfnoise_offset = upload(std::vector<float>(n_pixels, 0.F));
    selfnoise_raw = upload(std::vector<float>(n_pixels, 1.F));
    dde_noise = upload(std::vector<float>(n_pixels, 0.F));
    accept_mask = upload(std::vector<bool>(n_pixels * n_channels, true));
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
        .band_stop = 1,
        .boxsize = 1,
        .inner_boxsize = 0,
        .beam_halfsize = 0,
        .histogram_detail = 0,
        .max_sigma_bin = 2,
        .selfnoise_scale_factor = 1.F,
        .inv_band_size = 1.F,
        .ddenoise_scaling_factor = 1.F,
        .reject_noise_threshold = 0.F,
        .threshold = 2.F,
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

TEST_CASE("gpu_process_bands runs one job per stream", "[gpu][band_scheduler]")
{
  JobFixture first;
  JobFixture second;

  std::vector<GpuBandJob> jobs;
  jobs.push_back(std::move(*first.job));
  jobs.push_back(std::move(*second.job));

  gpu_process_bands(jobs);

  for (const auto &job : jobs)
  {
    CHECK(job.error == nullptr);
    CHECK(job.status == GpuBandStatus::no_detections);
    CHECK(job.workspace->plane.empty() == false);
    CHECK(job.result->median_noise.empty() == false);
  }

}

TEST_CASE("gpu_process_bands records per-job failures", "[gpu][band_scheduler]")
{
  JobFixture good;
  JobFixture bad;
  bad.job->input.sizes.cube = 0;

  std::vector<GpuBandJob> jobs;
  jobs.push_back(std::move(*good.job));
  jobs.push_back(std::move(*bad.job));

  gpu_process_bands(jobs);

  CHECK(jobs[0].error == nullptr);
  CHECK(jobs[0].status == GpuBandStatus::no_detections);
  CHECK(jobs[1].error != nullptr);

}

TEST_CASE("copy_band_jobs copies completed band results", "[gpu][band_scheduler]")
{
  JobFixture first;
  JobFixture second;

  std::vector<GpuBandJob> jobs;
  jobs.push_back(std::move(*first.job));
  jobs.push_back(std::move(*second.job));

  gpu_process_bands(jobs);

  std::vector<GpuBandHostResult> results;
  copy_band_jobs(jobs, results);

  REQUIRE(results.size() == jobs.size());
  for (const auto &result : results)
  {
    CHECK(result.label_count == 0U);
    CHECK_THAT(result.median_noise, WithinRel(1.F, 1e-6));
    CHECK_THAT(result.global_mad, WithinRel(1.4826F, 1e-6));
    CHECK(result.histogram0.size() == JobFixture::histogram_bins);
    CHECK(result.histogram1.size() == JobFixture::histogram_bins);
    CHECK(result.histogram2.size() == JobFixture::histogram_bins);
  }

}

TEST_CASE("gpu_process_plane assembles completed band jobs", "[gpu][plane]")
{
  JobFixture first;
  JobFixture second;

  std::vector<GpuBandJob> jobs;
  jobs.push_back(std::move(*first.job));
  jobs.push_back(std::move(*second.job));

  GpuPlaneHostResult result;
  gpu_process_plane(jobs, result);

  REQUIRE(result.statuses.size() == jobs.size());
  CHECK(result.statuses[0] == GpuBandStatus::no_detections);
  CHECK(result.statuses[1] == GpuBandStatus::no_detections);
  REQUIRE(result.bands.size() == jobs.size());
  CHECK(result.bands[0].label_count == 0U);
  CHECK(result.histograms[0].size() == JobFixture::histogram_bins);
  CHECK(result.histograms[1].size() == JobFixture::histogram_bins);
  CHECK(result.histograms[2].size() == JobFixture::histogram_bins);
  CHECK(result.median_noises.size() == jobs.size());
  for (const double median_noise : result.median_noises)
  {
    CHECK_THAT(median_noise, WithinRel(1.0, 1e-6));
  }
  REQUIRE(result.min_sigma.has_value() == true);
  REQUIRE(result.max_sigma.has_value() == true);
  CHECK_THAT(*result.min_sigma, WithinRel(1.F, 1e-6));
  CHECK_THAT(*result.max_sigma, WithinRel(1.F, 1e-6));

}

TEST_CASE("gpu_process_plane rethrows band failures", "[gpu][plane]")
{
  JobFixture good;
  JobFixture bad;
  bad.job->input.sizes.cube = 0;

  std::vector<GpuBandJob> jobs;
  jobs.push_back(std::move(*good.job));
  jobs.push_back(std::move(*bad.job));

  GpuPlaneHostResult result;
  CHECK_THROWS_AS(gpu_process_plane(jobs, result), std::runtime_error);

}
