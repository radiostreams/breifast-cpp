#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "gpu_band.hpp"
#include "gpu_test_utils.hpp"

namespace process_band_tests
{
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
    accept_mask = upload(std::vector<bool>(n_pixels, accept_value));
    band.accept_mask = accept_mask.data();

    plane = GpuPlaneInput{
        .cube = cube.data(),
        .weights = weights.data(),
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
  fixture.band.accept_mask = fixture.accept_mask.data();
}
} // namespace

TEST_CASE("gpu_process_band runs the RMS noise stage and reports no detections", "[gpu][process_band]")
{
  Fixture fixture(true, true);
  gpu_process_band(fixture.plane, fixture.band, fixture.workspace, fixture.result, fixture.stream);
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
  gpu_process_band(fixture.plane, fixture.band, fixture.workspace, fixture.result, fixture.stream);
  CHECK(fixture.result.label_count == 0U);

  CHECK_THAT(download_scalar(fixture.result.global_mad.data(), fixture.stream), WithinRel(1.4826F, 1e-6));
  CHECK_THAT(download_scalar(fixture.result.median_noise.data(), fixture.stream), WithinRel(1.4826F, 1e-6));
}

TEST_CASE("gpu_process_band reaches the detection stage", "[gpu][process_band]")
{
  Fixture fixture(true, true);
  fixture.plane.threshold = 0.5F;

  std::vector<bool> mask(Fixture::n_pixels, false);
  mask[0] = true;
  set_accept_mask(fixture, mask);

  gpu_process_band(fixture.plane, fixture.band, fixture.workspace, fixture.result, fixture.stream);
  CHECK(fixture.result.label_count == 1U);
}
} // namespace process_band_tests

namespace band_workspace_tests
{
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
} // namespace band_workspace_tests

namespace band_result_tests
{
using breifast::gpu::copy_band_result;
using breifast::gpu::GpuBandHostResult;
using breifast::gpu::GpuBandResult;
using breifast::gpu::Stream;
using breifast::gpu::test::upload;
using Catch::Matchers::Equals;

TEST_CASE("copy_band_result copies per-island arrays, histograms and scalars", "[gpu][band_result]")
{
  constexpr std::size_t histogram_bins = 3;
  constexpr unsigned int label_count = 2;

  GpuBandResult source;
  source.ensure(label_count, histogram_bins);
  source.label_count = label_count;

  const std::vector<float> floats = {1.F, 2.F};
  const std::vector<long> coordinates = {3L, 4L};
  const std::vector<unsigned int> sizes = {5U, 6U};
  const std::vector<unsigned long long> histogram0 = {10ULL, 11ULL, 12ULL};
  const std::vector<unsigned long long> histogram1 = {20ULL, 21ULL, 22ULL};
  const std::vector<unsigned long long> histogram2 = {30ULL, 31ULL, 32ULL};

  source.islands.peak_value = upload(floats);
  source.islands.peak_local_noise = upload(floats);
  source.islands.peak_y = upload(coordinates);
  source.islands.peak_x = upload(coordinates);
  source.islands.island_size = upload(sizes);
  source.islands.stamp_extreme = upload(floats);
  source.islands.donut_std = upload(floats);
  source.islands.donut_lmad = upload(floats);
  source.islands.beam_chi2 = upload(floats);
  source.islands.flux = upload(floats);
  source.islands.boost = upload(floats);
  source.islands.noise_at_peak = upload(floats);
  source.islands.rms_at_peak = upload(floats);
  source.islands.selfnoise_at_peak = upload(floats);
  source.histogram0 = upload(histogram0);
  source.histogram1 = upload(histogram1);
  source.histogram2 = upload(histogram2);

  const float min_sigma = 0.1F;
  const float max_sigma = 0.2F;
  const float median_noise = 0.3F;
  const float rms_floor = 0.4F;
  const float global_mad = 0.5F;
  source.min_sigma = upload(std::vector<float>{min_sigma});
  source.max_sigma = upload(std::vector<float>{max_sigma});
  source.median_noise = upload(std::vector<float>{median_noise});
  source.rms_floor = upload(std::vector<float>{rms_floor});
  source.global_mad = upload(std::vector<float>{global_mad});

  GpuBandHostResult destination;
  const Stream stream;
  copy_band_result(source, destination, stream);

  CHECK(destination.label_count == label_count);
  CHECK_THAT(destination.islands.peak_value, Equals(floats));
  CHECK_THAT(destination.islands.peak_local_noise, Equals(floats));
  CHECK_THAT(destination.islands.peak_y, Equals(coordinates));
  CHECK_THAT(destination.islands.peak_x, Equals(coordinates));
  CHECK_THAT(destination.islands.island_size, Equals(sizes));
  CHECK_THAT(destination.islands.stamp_extreme, Equals(floats));
  CHECK_THAT(destination.islands.donut_std, Equals(floats));
  CHECK_THAT(destination.islands.donut_lmad, Equals(floats));
  CHECK_THAT(destination.islands.beam_chi2, Equals(floats));
  CHECK_THAT(destination.islands.flux, Equals(floats));
  CHECK_THAT(destination.islands.boost, Equals(floats));
  CHECK_THAT(destination.islands.noise_at_peak, Equals(floats));
  CHECK_THAT(destination.islands.rms_at_peak, Equals(floats));
  CHECK_THAT(destination.islands.selfnoise_at_peak, Equals(floats));
  CHECK_THAT(destination.histogram0, Equals(histogram0));
  CHECK_THAT(destination.histogram1, Equals(histogram1));
  CHECK_THAT(destination.histogram2, Equals(histogram2));
  CHECK(destination.min_sigma == min_sigma);
  CHECK(destination.max_sigma == max_sigma);
  CHECK(destination.median_noise == median_noise);
  CHECK(destination.rms_floor == rms_floor);
  CHECK(destination.global_mad == global_mad);
}
} // namespace band_result_tests
