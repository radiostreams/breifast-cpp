#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_band_result.cuh"
#include "gpu_buffers.hpp"
#include "gpu_plane_detections.hpp"
#include "gpu_process_band.cuh"

using breifast::DetectionArrays;
using breifast::gpu::assemble_plane_bands;
using breifast::gpu::assemble_plane_detections;
using breifast::gpu::copy_band_jobs;
using breifast::gpu::destroy_band_job;
using breifast::gpu::DeviceVector;
using breifast::gpu::GpuBandHostResult;
using breifast::gpu::GpuBandJob;
using breifast::gpu::GpuBandLabels;
using breifast::gpu::GpuBandStatus;
using breifast::gpu::gpu_process_plane;
using breifast::gpu::GpuPlaneHostResult;
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
  DeviceVector<uint8_t> accept_mask;
  DeviceVector<uint8_t> nonzero_channels;

  GpuBandJob job;

  DetectionFixture()
  {
    cube.assign(n_pixels * n_channels, 1.F);
    weights.assign(n_channels, 1.F);
    selfnoise_offset.assign(n_pixels, 0.F);
    selfnoise_raw.assign(n_pixels, 1.F);
    dde_noise.assign(n_pixels, 0.F);
    accept_mask.assign(n_pixels * n_channels, 0U);
    accept_mask[0] = 1U;
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
    job.input.band_stop = 2;
    job.input.boxsize = 1;
    job.input.inner_boxsize = 0;
    job.input.beam_halfsize = 0;
    job.input.histogram_detail = 1;
    job.input.max_sigma_bin = 5;
    job.input.selfnoise_scale_factor = 1.F;
    job.input.inv_band_size = 0.5F;
    job.input.inv_weights_sum = 0.5F;
    job.input.ddenoise_scaling_factor = 1.F;
    job.input.reject_noise_threshold = 0.F;
    job.input.threshold = 0.5F;
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

TEST_CASE("GPU plane job produces an assembled detection row", "[gpu][plane_detections][e2e]")
{
  DetectionFixture fixture;
  std::vector<GpuBandJob> jobs;
  jobs.push_back(fixture.job);

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

  for (auto &job : jobs)
  {
    destroy_band_job(job);
  }
}
