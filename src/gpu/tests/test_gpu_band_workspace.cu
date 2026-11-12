#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <tuple>

#include "gpu_band.hpp"

using breifast::gpu::GpuBandResult;
using breifast::gpu::GpuBandWorkspace;
using breifast::gpu::GpuPlaneInput;

namespace
{
// A 64 x 128 plane. ensure only reads the sizes, so the pointers stay null.
constexpr std::size_t n_pixels = 64 * 128;

GpuPlaneInput make_plane()
{
  return GpuPlaneInput{.n_y = 64, .n_x = 128, .n_channels = 1, .boxsize = 5, .max_sigma_bin = 5};
}

template <typename Columns> bool all_columns_hold(const Columns &columns, std::size_t size)
{
  return std::apply([size](const auto &...column) { return ((column.size() == size) && ...); }, columns.tie());
}
} // namespace

TEST_CASE("GpuBandWorkspace grows every buffer to what the plane needs", "[gpu][band_workspace]")
{
  GpuBandWorkspace workspace;
  workspace.ensure(make_plane());

  CHECK(workspace.plane.size() == n_pixels);
  CHECK(workspace.noise_map.size() == n_pixels);
  CHECK(workspace.snr.size() == n_pixels);
  CHECK(workspace.mask.size() == n_pixels);
  CHECK(workspace.row_sums.size() == n_pixels);
  CHECK(workspace.rms_map.size() == n_pixels);
  CHECK(workspace.labels.size() == n_pixels);
  CHECK(workspace.label_scratch.root_ranks.size() == n_pixels);
  CHECK(workspace.label_scratch.scan_storage.empty() == false);
  CHECK(workspace.any_flag.size() == 1);
  CHECK(workspace.label_count.size() == 1);
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
  GpuBandWorkspace workspace;
  workspace.ensure(make_plane());
  workspace.ensure(GpuPlaneInput{.n_y = 8, .n_x = 8, .n_channels = 1, .boxsize = 1, .max_sigma_bin = 5});
  CHECK(workspace.plane.size() == n_pixels);

  GpuBandResult result;
  result.ensure(64, 10);
  result.ensure(8, 4);
  CHECK(result.histogram_bins == 4); // the valid count follows the band, the buffers do not shrink
  CHECK(all_columns_hold(result.islands, 64));
  CHECK(result.histogram0.size() == 10);
}
