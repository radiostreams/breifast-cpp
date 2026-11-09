#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_buffers.hpp"
#include "gpu_process_band.cuh"
#include "gpu_runtime.hpp"

using breifast::gpu::create_band_result;
using breifast::gpu::create_band_workspace;
using breifast::gpu::destroy_band_job;
using breifast::gpu::DeviceVector;
using breifast::gpu::GpuBandJob;
using breifast::gpu::GpuBandStatus;
using breifast::gpu::gpu_process_bands;

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
  DeviceVector<uint8_t> accept_mask;
  DeviceVector<uint8_t> nonzero_channels;

  GpuBandJob job;

  JobFixture()
  {
    cube.assign(n_pixels * n_channels, 1.F);
    weights.assign(n_channels, 1.F);
    selfnoise_offset.assign(n_pixels, 0.F);
    selfnoise_raw.assign(n_pixels, 1.F);
    dde_noise.assign(n_pixels, 0.F);
    accept_mask.assign(n_pixels * n_channels, 1U);
    nonzero_channels.assign(n_channels, 1U);

    job.input.sizes.cube = n_pixels * n_channels;
    job.input.sizes.weights = n_channels;
    job.input.sizes.nonzero_channels = n_channels;
    job.input.sizes.accept_mask = n_pixels * n_channels;
    job.input.sizes.selfnoise_offset = n_pixels;
    job.input.sizes.selfnoise_raw = n_pixels;
    job.input.sizes.dde_noise = n_pixels;

    job.input.cube = thrust::raw_pointer_cast(cube.data());
    job.input.weights = thrust::raw_pointer_cast(weights.data());
    job.input.nonzero_channels = reinterpret_cast<const bool *>(thrust::raw_pointer_cast(nonzero_channels.data()));
    job.input.accept_mask = reinterpret_cast<const bool *>(thrust::raw_pointer_cast(accept_mask.data()));
    job.input.selfnoise_offset = thrust::raw_pointer_cast(selfnoise_offset.data());
    job.input.selfnoise_raw = thrust::raw_pointer_cast(selfnoise_raw.data());
    job.input.dde_noise = thrust::raw_pointer_cast(dde_noise.data());

    job.input.n_y = n_y;
    job.input.n_x = n_x;
    job.input.n_pixels = n_pixels;
    job.input.n_channels = n_channels;
    job.input.band_start = 0;
    job.input.band_stop = 1;
    job.input.boxsize = 1;
    job.input.inner_boxsize = 0;
    job.input.beam_halfsize = 0;
    job.input.histogram_detail = 0;
    job.input.max_sigma_bin = 2;
    job.input.selfnoise_scale_factor = 1.F;
    job.input.inv_band_size = 1.F;
    job.input.inv_weights_sum = 1.F;
    job.input.ddenoise_scaling_factor = 1.F;
    job.input.reject_noise_threshold = 0.F;
    job.input.threshold = 2.F;
    job.input.max_to_std_ratio_value = 1.F;
    job.input.use_rms_noise_estimator = true;
    job.input.subtract_mean = false;

    job.workspace_sizes.n_pixels = n_pixels;
    job.workspace_sizes.stamp_capacity = 1;
    job.workspace_sizes.stamp_pixels = 1;
    job.workspace_sizes.beam_stamp_pixels = 0;
    job.workspace_sizes.histogram_bins = histogram_bins;
    job.result_n_pixels = n_pixels;
    job.result_histogram_bins = histogram_bins;
  }
};
} // namespace

TEST_CASE("gpu_process_bands runs one job per stream", "[gpu][band_scheduler]")
{
  JobFixture first;
  JobFixture second;

  std::vector<GpuBandJob> jobs;
  jobs.push_back(first.job);
  jobs.push_back(second.job);

  gpu_process_bands(jobs);

  for (const auto &job : jobs)
  {
    CHECK(job.error == nullptr);
    CHECK(job.status == GpuBandStatus::no_detections);
    CHECK(job.workspace.plane != nullptr);
    CHECK(job.result.median_noise != nullptr);
  }

  for (auto &job : jobs)
  {
    destroy_band_job(job);
  }
}

TEST_CASE("gpu_process_bands records per-job failures", "[gpu][band_scheduler]")
{
  JobFixture good;
  JobFixture bad;
  bad.job.input.sizes.cube = 0;

  std::vector<GpuBandJob> jobs;
  jobs.push_back(good.job);
  jobs.push_back(bad.job);

  gpu_process_bands(jobs);

  CHECK(jobs[0].error == nullptr);
  CHECK(jobs[0].status == GpuBandStatus::no_detections);
  CHECK(jobs[1].error != nullptr);

  for (auto &job : jobs)
  {
    destroy_band_job(job);
  }
}
