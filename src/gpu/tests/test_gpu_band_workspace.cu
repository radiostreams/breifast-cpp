#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <tuple>

#include "gpu_band.hpp"

using breifast::gpu::GpuBand;
using breifast::gpu::GpuBandResult;
using breifast::gpu::GpuBandWorkspace;
using breifast::gpu::GpuPlaneInput;

namespace
{
// A 64 x 128 plane with 5 x 5 stamps and 100 islands. ensure and ensure_islands only read the sizes, so the pointers
// stay null.
constexpr std::size_t n_pixels = 64 * 128;
constexpr std::size_t n_islands = 100;

GpuPlaneInput make_plane()
{
  return GpuPlaneInput{.n_y = 64, .n_x = 128, .n_channels = 1, .boxsize = 5, .max_sigma_bin = 5};
}

// A band with a 7 x 7 beam kernel. ensure_islands only tests whether the kernel pointer is null.
GpuBand make_band(const float *beam_kernel)
{
  return GpuBand{.start = 0, .stop = 1, .beam_kernel = beam_kernel, .beam_halfsize = 3};
}

template <typename Columns> bool all_columns_hold(const Columns &columns, std::size_t size)
{
  return std::apply([size](const auto &...column) { return ((column.size() == size) && ...); }, columns.tie());
}
} // namespace

TEST_CASE("GpuBandWorkspace grows every buffer to what the band needs", "[gpu][band_workspace]")
{
  const float kernel = 0.F;
  GpuBandWorkspace workspace;
  workspace.ensure(make_plane());
  workspace.ensure_islands(make_plane(), make_band(&kernel), n_islands);

  CHECK(workspace.plane.size() == n_pixels);
  CHECK(workspace.noise_map.size() == n_pixels);
  CHECK(workspace.snr.size() == n_pixels);
  CHECK(workspace.mask.size() == n_pixels);
  CHECK(workspace.row_sums.size() == n_pixels);
  CHECK(workspace.rms_map.size() == n_pixels);
  CHECK(workspace.labels.size() == n_pixels);
  CHECK(workspace.label_scratch.root_ranks.size() == n_pixels);
  CHECK(workspace.label_scratch.scan_storage.empty() == false);
  CHECK(workspace.island_scratch.value_keys.size() == n_islands);
  CHECK(workspace.donut_stamps.size() == n_islands * 25);
  CHECK(workspace.beam_stamps.size() == n_islands * 49);
  CHECK(workspace.any_flag.size() == 1);
  CHECK(workspace.label_count.size() == 1);
}

TEST_CASE("GpuBandWorkspace leaves the beam stamps empty without a beam kernel", "[gpu][band_workspace]")
{
  GpuBandWorkspace workspace;
  workspace.ensure_islands(make_plane(), make_band(nullptr), n_islands);
  CHECK(workspace.donut_stamps.empty() == false);
  CHECK(workspace.beam_stamps.empty() == true);
}

TEST_CASE("GpuBandResult grows every output buffer", "[gpu][band_workspace]")
{
  GpuBandResult result;
  result.ensure(64, 10);

  CHECK(result.histogram_bins == 10);
  CHECK(all_columns_hold(result.islands, 64));
  CHECK(result.histogram0.size() == 10);
  CHECK(result.histogram1.size() == 10);
  CHECK(result.histogram2.size() == 10);
  CHECK(result.min_sigma.size() == 1);
  CHECK(result.max_sigma.size() == 1);
  CHECK(result.median_noise.size() == 1);
  CHECK(result.rms_floor.size() == 1);
  CHECK(result.global_mad.size() == 1);
}

TEST_CASE("GpuBandWorkspace and GpuBandResult keep their buffers for a smaller band", "[gpu][band_workspace]")
{
  const float kernel = 0.F;
  GpuBandWorkspace workspace;
  workspace.ensure(make_plane());
  workspace.ensure_islands(make_plane(), make_band(&kernel), n_islands);
  const GpuPlaneInput small{.n_y = 8, .n_x = 8, .n_channels = 1, .boxsize = 1, .max_sigma_bin = 5};
  workspace.ensure(small);
  workspace.ensure_islands(small, make_band(nullptr), 10);
  CHECK(workspace.plane.size() == n_pixels);
  CHECK(workspace.donut_stamps.size() == n_islands * 25);
  CHECK(workspace.beam_stamps.size() == n_islands * 49);

  GpuBandResult result;
  result.ensure(64, 10);
  result.ensure(8, 4);
  CHECK(result.histogram_bins == 4); // the valid count follows the band, the buffers do not shrink
  CHECK(all_columns_hold(result.islands, 64));
  CHECK(result.histogram0.size() == 10);
}
