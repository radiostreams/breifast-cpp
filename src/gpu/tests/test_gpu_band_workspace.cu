#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "gpu_band_workspace.cuh"

using breifast::gpu::GpuBandResult;
using breifast::gpu::GpuBandWorkspace;
using breifast::gpu::GpuBandWorkspaceSizes;
using breifast::gpu::max_stamp_capacity;
using Catch::Matchers::ContainsSubstring;

namespace
{
GpuBandWorkspaceSizes make_sizes()
{
  return GpuBandWorkspaceSizes{
      .n_pixels = 1024,
      .stamp_capacity = 8,
      .stamp_pixels = 25,
      .beam_stamp_pixels = 49,
      .histogram_bins = 10,
  };
}
} // namespace

TEST_CASE("GpuBandWorkspace allocates every scratch buffer", "[gpu][band_workspace]")
{
  const auto sizes = make_sizes();
  const GpuBandWorkspace workspace(sizes);

  CHECK(workspace.sizes.n_pixels == sizes.n_pixels);
  CHECK(workspace.sizes.stamp_capacity == sizes.stamp_capacity);
  CHECK(workspace.sizes.stamp_pixels == sizes.stamp_pixels);
  CHECK(workspace.sizes.beam_stamp_pixels == sizes.beam_stamp_pixels);
  CHECK(workspace.sizes.histogram_bins == sizes.histogram_bins);

  CHECK(workspace.plane.size() == sizes.n_pixels);
  CHECK(workspace.noise_map.size() == sizes.n_pixels);
  CHECK(workspace.snr.size() == sizes.n_pixels);
  CHECK(workspace.mask.size() == sizes.n_pixels);
  CHECK(workspace.row_sums.size() == sizes.n_pixels);
  CHECK(workspace.rms_map.size() == sizes.n_pixels);
  CHECK(workspace.labels.size() == sizes.n_pixels);
  CHECK(workspace.label_scratch.size() == sizes.n_pixels);
  CHECK(workspace.donut_stamps.size() == sizes.stamp_capacity * sizes.stamp_pixels);
  CHECK(workspace.beam_stamps.size() == sizes.stamp_capacity * sizes.beam_stamp_pixels);
  CHECK(workspace.any_flag.size() == 1);
  CHECK(workspace.label_count.size() == 1);
}

TEST_CASE("GpuBandWorkspace skips beam stamps when beam is disabled", "[gpu][band_workspace]")
{
  auto sizes = make_sizes();
  sizes.beam_stamp_pixels = 0;

  const GpuBandWorkspace workspace(sizes);
  CHECK(workspace.donut_stamps.empty() == false);
  CHECK(workspace.beam_stamps.empty() == true);
}

TEST_CASE("GpuBandWorkspace rejects invalid sizes and the stamp cap", "[gpu][band_workspace]")
{
  const auto sizes = make_sizes();

  auto bad = sizes;
  bad.n_pixels = 0;
  CHECK_THROWS_WITH(GpuBandWorkspace(bad), ContainsSubstring("n_pixels"));

  bad = sizes;
  bad.stamp_pixels = 0;
  CHECK_THROWS_WITH(GpuBandWorkspace(bad), ContainsSubstring("stamp_pixels"));

  bad = sizes;
  bad.stamp_capacity = 0;
  CHECK_THROWS_WITH(GpuBandWorkspace(bad), ContainsSubstring("stamp_capacity"));

  bad = sizes;
  bad.stamp_capacity = max_stamp_capacity + 1;
  CHECK_THROWS_WITH(GpuBandWorkspace(bad), ContainsSubstring("exceeds the maximum"));
}

TEST_CASE("GpuBandResult allocates every output buffer", "[gpu][band_workspace]")
{
  constexpr std::size_t n_pixels = 64;
  constexpr std::size_t histogram_bins = 10;

  const GpuBandResult result(n_pixels, histogram_bins);

  CHECK(result.n_pixels == n_pixels);
  CHECK(result.histogram_bins == histogram_bins);
  CHECK(result.peak_value.size() == n_pixels);
  CHECK(result.peak_local_noise.size() == n_pixels);
  CHECK(result.peak_y.size() == n_pixels);
  CHECK(result.peak_x.size() == n_pixels);
  CHECK(result.island_size.size() == n_pixels);
  CHECK(result.stamp_extreme.size() == n_pixels);
  CHECK(result.donut_std.size() == n_pixels);
  CHECK(result.donut_lmad.size() == n_pixels);
  CHECK(result.beam_chi2.size() == n_pixels);
  CHECK(result.flux.size() == n_pixels);
  CHECK(result.boost.size() == n_pixels);
  CHECK(result.boost_found.size() == n_pixels);
  CHECK(result.noise_at_peak.size() == n_pixels);
  CHECK(result.rms_at_peak.size() == n_pixels);
  CHECK(result.selfnoise_at_peak.size() == n_pixels);
  CHECK(result.histogram0.size() == histogram_bins);
  CHECK(result.histogram1.size() == histogram_bins);
  CHECK(result.histogram2.size() == histogram_bins);
  CHECK(result.min_sigma.size() == 1);
  CHECK(result.max_sigma.size() == 1);
  CHECK(result.median_noise.size() == 1);
  CHECK(result.rms_floor.size() == 1);
  CHECK(result.global_mad.size() == 1);
}

TEST_CASE("GpuBandResult rejects empty sizes", "[gpu][band_workspace]")
{
  CHECK_THROWS_WITH(GpuBandResult(0, 10), ContainsSubstring("n_pixels"));
  CHECK_THROWS_WITH(GpuBandResult(10, 0), ContainsSubstring("histogram_bins"));
}
