#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cuda_runtime.h>

#include "gpu_band_workspace.cuh"
#include "gpu_runtime.hpp"

using breifast::gpu::create_band_result;
using breifast::gpu::create_band_workspace;
using breifast::gpu::destroy_band_result;
using breifast::gpu::destroy_band_workspace;
using breifast::gpu::GpuBandResult;
using breifast::gpu::GpuBandWorkspace;
using breifast::gpu::GpuBandWorkspaceSizes;
using breifast::gpu::max_stamp_capacity;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("create_band_workspace allocates every scratch buffer", "[gpu][band_workspace]")
{
  GpuBandWorkspaceSizes sizes;
  sizes.n_pixels = 1024;
  sizes.stamp_capacity = 8;
  sizes.stamp_pixels = 25;
  sizes.beam_stamp_pixels = 49;
  sizes.histogram_bins = 10;

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  auto workspace = create_band_workspace(sizes, stream);

  CHECK(workspace.sizes.n_pixels == sizes.n_pixels);
  CHECK(workspace.sizes.stamp_capacity == sizes.stamp_capacity);
  CHECK(workspace.sizes.stamp_pixels == sizes.stamp_pixels);
  CHECK(workspace.sizes.beam_stamp_pixels == sizes.beam_stamp_pixels);
  CHECK(workspace.sizes.histogram_bins == sizes.histogram_bins);

  CHECK(workspace.plane != nullptr);
  CHECK(workspace.noise_map != nullptr);
  CHECK(workspace.snr != nullptr);
  CHECK(workspace.mask != nullptr);
  CHECK(workspace.row_sums != nullptr);
  CHECK(workspace.rms_map != nullptr);
  CHECK(workspace.labels != nullptr);
  CHECK(workspace.label_scratch != nullptr);
  CHECK(workspace.donut_stamps != nullptr);
  CHECK(workspace.beam_stamps != nullptr);
  CHECK(workspace.any_flag != nullptr);
  CHECK(workspace.label_count != nullptr);
  CHECK(workspace.adjacent_noise != nullptr);

  destroy_band_workspace(workspace, stream);
  gpuStreamSynchronize(stream);

  CHECK(workspace.plane == nullptr);
  CHECK(workspace.noise_map == nullptr);
  CHECK(workspace.snr == nullptr);
  CHECK(workspace.mask == nullptr);
  CHECK(workspace.row_sums == nullptr);
  CHECK(workspace.rms_map == nullptr);
  CHECK(workspace.labels == nullptr);
  CHECK(workspace.label_scratch == nullptr);
  CHECK(workspace.donut_stamps == nullptr);
  CHECK(workspace.beam_stamps == nullptr);
  CHECK(workspace.any_flag == nullptr);
  CHECK(workspace.label_count == nullptr);
  CHECK(workspace.adjacent_noise == nullptr);
  CHECK(workspace.sizes.n_pixels == 0);

  gpuStreamDestroy(stream);
}

TEST_CASE("create_band_workspace skips beam stamps when beam is disabled", "[gpu][band_workspace]")
{
  GpuBandWorkspaceSizes sizes;
  sizes.n_pixels = 64;
  sizes.stamp_capacity = 2;
  sizes.stamp_pixels = 9;
  sizes.beam_stamp_pixels = 0;
  sizes.histogram_bins = 4;

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  auto workspace = create_band_workspace(sizes, stream);
  CHECK(workspace.donut_stamps != nullptr);
  CHECK(workspace.beam_stamps == nullptr);

  destroy_band_workspace(workspace, stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);
}

TEST_CASE("create_band_workspace rejects invalid sizes and the stamp cap", "[gpu][band_workspace]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  GpuBandWorkspaceSizes sizes;
  sizes.n_pixels = 64;
  sizes.stamp_capacity = 2;
  sizes.stamp_pixels = 9;
  sizes.histogram_bins = 4;

  auto bad = sizes;
  bad.n_pixels = 0;
  CHECK_THROWS_WITH(create_band_workspace(bad, stream), ContainsSubstring("n_pixels"));

  bad = sizes;
  bad.stamp_pixels = 0;
  CHECK_THROWS_WITH(create_band_workspace(bad, stream), ContainsSubstring("stamp_pixels"));

  bad = sizes;
  bad.stamp_capacity = 0;
  CHECK_THROWS_WITH(create_band_workspace(bad, stream), ContainsSubstring("stamp_capacity"));

  bad = sizes;
  bad.stamp_capacity = max_stamp_capacity + 1;
  CHECK_THROWS_WITH(create_band_workspace(bad, stream), ContainsSubstring("exceeds the maximum"));

  gpuStreamDestroy(stream);
}

TEST_CASE("create_band_result allocates every output buffer", "[gpu][band_workspace]")
{
  constexpr std::size_t n_pixels = 64;
  constexpr std::size_t histogram_bins = 10;

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  auto result = create_band_result(n_pixels, histogram_bins, stream);

  CHECK(result.n_pixels == n_pixels);
  CHECK(result.histogram_bins == histogram_bins);
  CHECK(result.peak_value != nullptr);
  CHECK(result.peak_local_noise != nullptr);
  CHECK(result.peak_y != nullptr);
  CHECK(result.peak_x != nullptr);
  CHECK(result.island_size != nullptr);
  CHECK(result.stamp_extreme != nullptr);
  CHECK(result.donut_std != nullptr);
  CHECK(result.donut_lmad != nullptr);
  CHECK(result.beam_chi2 != nullptr);
  CHECK(result.flux != nullptr);
  CHECK(result.boost != nullptr);
  CHECK(result.boost_found != nullptr);
  CHECK(result.noise_at_peak != nullptr);
  CHECK(result.rms_at_peak != nullptr);
  CHECK(result.selfnoise_at_peak != nullptr);
  CHECK(result.histogram0 != nullptr);
  CHECK(result.histogram1 != nullptr);
  CHECK(result.histogram2 != nullptr);
  CHECK(result.min_sigma != nullptr);
  CHECK(result.max_sigma != nullptr);
  CHECK(result.median_noise != nullptr);
  CHECK(result.rms_floor != nullptr);
  CHECK(result.global_mad != nullptr);

  destroy_band_result(result, stream);
  gpuStreamSynchronize(stream);

  CHECK(result.peak_value == nullptr);
  CHECK(result.histogram2 == nullptr);
  CHECK(result.global_mad == nullptr);
  CHECK(result.n_pixels == 0);

  gpuStreamDestroy(stream);
}

TEST_CASE("create_band_result rejects empty sizes", "[gpu][band_workspace]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  CHECK_THROWS_WITH(create_band_result(0, 10, stream), ContainsSubstring("n_pixels"));
  CHECK_THROWS_WITH(create_band_result(10, 0, stream), ContainsSubstring("histogram_bins"));

  gpuStreamDestroy(stream);
}
