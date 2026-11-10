#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <cmath>
#include <cstddef>
#include <optional>

#include "gpu_lot_integration.hpp"
#include "gpu_plane_integration.hpp"

using namespace breifast;
using Catch::Matchers::Equals;
using Catch::Matchers::WithinAbs;

namespace
{
constexpr std::size_t n_y = 3;
constexpr std::size_t n_x = 4;
constexpr std::size_t n_channels = 2;

struct Fixture
{
  PlaneBatchItem item;
  PlaneBatchGroup group;
  PlaneConfig config;
};

Fixture make_fixture(const bool use_rms, const long histogram_detail)
{
  Fixture fixture;
  fixture.item.stokes_cube.resize({n_channels, n_y, n_x});
  fixture.item.stokes_cube.fill(1.F);
  fixture.item.weights = xt::xarray<float>({1.F, 1.F});
  fixture.item.nonzero_channels = xt::xarray<bool>({true, true});
  fixture.item.stokes_label = "I";

  fixture.group.accept_mask.resize({n_channels, n_y, n_x});
  fixture.group.accept_mask.fill(false);
  fixture.group.accept_mask(0, 0, 0) = true;
  fixture.group.selfnoise_offset.resize({n_y, n_x});
  fixture.group.selfnoise_offset.fill(0.F);
  fixture.group.selfnoise_raw.resize({n_y, n_x});
  fixture.group.selfnoise_raw.fill(1.F);
  fixture.group.dde_noise.resize({n_y, n_x});
  fixture.group.dde_noise.fill(0.F);

  fixture.config.subtract_mean = false;
  fixture.config.use_rms_noise_estimator = use_rms;
  fixture.config.boxsize = 3;
  fixture.config.inner_boxsize = 0;
  fixture.config.histogram_detail = histogram_detail;
  fixture.config.max_sigma_bin = 5;
  fixture.config.threshold = 0.5F;
  fixture.config.beam_size_enabled = false;
  fixture.config.band_slices = {BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "band"}};
  return fixture;
}

void check_close(const double actual, const double expected, const char *name)
{
  INFO(name);
  if (std::isnan(actual) && std::isnan(expected))
  {
    return;
  }
  CHECK_THAT(actual, WithinAbs(expected, 1e-5));
}


void compare_plane_results(const ProcessPlaneResult &cpu, const ProcessPlaneResult &gpu)
{
  CHECK(cpu.any_data == gpu.any_data);
  CHECK_THAT(gpu.histograms[0], Equals(cpu.histograms[0]));
  CHECK_THAT(gpu.histograms[1], Equals(cpu.histograms[1]));
  CHECK_THAT(gpu.histograms[2], Equals(cpu.histograms[2]));

  REQUIRE(cpu.min_sigma.has_value() == gpu.min_sigma.has_value());
  REQUIRE(cpu.max_sigma.has_value() == gpu.max_sigma.has_value());
  if (cpu.min_sigma.has_value())
  {
    check_close(*gpu.min_sigma, *cpu.min_sigma, "min_sigma");
    check_close(*gpu.max_sigma, *cpu.max_sigma, "max_sigma");
  }

  REQUIRE(cpu.median_noises.size() == gpu.median_noises.size());
  for (std::size_t i = 0; i < cpu.median_noises.size(); ++i)
  {
    check_close(gpu.median_noises[i], cpu.median_noises[i], "median_noise");
  }

  const auto &cpu_detections = cpu.detections;
  const auto &gpu_detections = gpu.detections;
  REQUIRE(cpu_detections.size() == gpu_detections.size());
  for (std::size_t i = 0; i < cpu_detections.size(); ++i)
  {
    CHECK(gpu_detections.stokes_label()[i] == cpu_detections.stokes_label()[i]);
    CHECK(gpu_detections.band_label()[i] == cpu_detections.band_label()[i]);
    CHECK(gpu_detections.peak_y()[i] == cpu_detections.peak_y()[i]);
    CHECK(gpu_detections.peak_x()[i] == cpu_detections.peak_x()[i]);
    CHECK(gpu_detections.island_size()[i] == cpu_detections.island_size()[i]);
    check_close(gpu_detections.flux()[i], cpu_detections.flux()[i], "flux");
    check_close(gpu_detections.local_noise()[i], cpu_detections.local_noise()[i], "local_noise");
    check_close(gpu_detections.local_noise_median_ratio()[i], cpu_detections.local_noise_median_ratio()[i],
                "local_noise_median_ratio");
    check_close(gpu_detections.signal_to_noise()[i], cpu_detections.signal_to_noise()[i], "signal_to_noise");
    check_close(gpu_detections.signal_to_noise_rms()[i], cpu_detections.signal_to_noise_rms()[i],
                "signal_to_noise_rms");
    check_close(gpu_detections.signal_to_noise_extended_box()[i], cpu_detections.signal_to_noise_extended_box()[i],
                "signal_to_noise_extended_box");
    check_close(gpu_detections.signal_to_noise_local_mad()[i], cpu_detections.signal_to_noise_local_mad()[i],
                "signal_to_noise_local_mad");
    check_close(gpu_detections.signal_to_noise_global_mad()[i], cpu_detections.signal_to_noise_global_mad()[i],
                "signal_to_noise_global_mad");
    check_close(gpu_detections.self_noise()[i], cpu_detections.self_noise()[i], "self_noise");
    check_close(gpu_detections.excess_noise_factor()[i], cpu_detections.excess_noise_factor()[i], "excess_noise_factor");
    check_close(gpu_detections.beam_chi2()[i], cpu_detections.beam_chi2()[i], "beam_chi2");
    check_close(gpu_detections.boost()[i], cpu_detections.boost()[i], "boost");
  }

  REQUIRE(cpu.variance_stats.has_value() == gpu.variance_stats.has_value());
  if (cpu.variance_stats.has_value())
  {
    const auto &cpu_variance = *cpu.variance_stats;
    const auto &gpu_variance = *gpu.variance_stats;
    REQUIRE(cpu_variance.sum_pow2.shape() == gpu_variance.sum_pow2.shape());
    for (std::size_t i = 0; i < cpu_variance.sum_pow2.size(); ++i)
    {
      check_close(gpu_variance.sum_pow2.data()[i], cpu_variance.sum_pow2.data()[i], "sum_pow2");
      check_close(gpu_variance.sum_pow3.data()[i], cpu_variance.sum_pow3.data()[i], "sum_pow3");
      check_close(gpu_variance.sum_pow4.data()[i], cpu_variance.sum_pow4.data()[i], "sum_pow4");
    }
    REQUIRE(cpu_variance.nonzero_channels_mask.shape() == gpu_variance.nonzero_channels_mask.shape());
    for (std::size_t i = 0; i < cpu_variance.nonzero_channels_mask.size(); ++i)
    {
      CHECK(gpu_variance.nonzero_channels_mask.data()[i] == cpu_variance.nonzero_channels_mask.data()[i]);
    }
  }
}
void run_parity(Fixture fixture)
{
  auto cpu_item = fixture.item;
  auto gpu_item = fixture.item;
  ProcessBandScratch scratch;
  const auto cpu_result = process_plane(scratch, fixture.config, cpu_item, fixture.group);
  const auto gpu_result = gpu::gpu_process_plane_batch(gpu_item, fixture.group, fixture.config);
  compare_plane_results(cpu_result, gpu_result);
}
} // namespace

TEST_CASE("GPU plane result matches CPU in RMS noise mode", "[gpu][parity]")
{
  auto fixture = make_fixture(true, 1);
  auto cpu_item = fixture.item;
  auto gpu_item = fixture.item;
  ProcessBandScratch scratch;
  const auto cpu_result = process_plane(scratch, fixture.config, cpu_item, fixture.group);
  const auto gpu_result = gpu::gpu_process_plane_batch(gpu_item, fixture.group, fixture.config);
  compare_plane_results(cpu_result, gpu_result);
}

TEST_CASE("GPU plane result matches CPU in global-MAD noise mode", "[gpu][parity]")
{
  auto fixture = make_fixture(false, 2);
  auto cpu_item = fixture.item;
  auto gpu_item = fixture.item;
  ProcessBandScratch scratch;
  const auto cpu_result = process_plane(scratch, fixture.config, cpu_item, fixture.group);
  const auto gpu_result = gpu::gpu_process_plane_batch(gpu_item, fixture.group, fixture.config);
  compare_plane_results(cpu_result, gpu_result);
}

TEST_CASE("GPU parity for multiple islands", "[gpu][parity]")
{
  auto fixture = make_fixture(true, 1);
  fixture.group.accept_mask.fill(false);
  fixture.group.accept_mask(0, 0, 0) = true;
  fixture.group.accept_mask(0, 0, 2) = true;
  run_parity(fixture);
}

TEST_CASE("GPU parity for multiple bands in reverse order", "[gpu][parity]")
{
  auto fixture = make_fixture(true, 1);
  fixture.config.band_slices = {
      BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"},
      BandSlice{.start = 1, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B1"},
      BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B0~1"},
  };
  run_parity(fixture);
}

TEST_CASE("GPU parity with a fixed beam kernel", "[gpu][parity]")
{
  auto fixture = make_fixture(true, 1);
  fixture.config.beam_size_enabled = true;
  fixture.config.fixed_beam_halfsize = 1;
  fixture.config.fixed_beam_kernel = xt::xtensor<double, 2>::from_shape({3, 3});
  fixture.config.fixed_beam_kernel->fill(1.0 / 9.0);
  run_parity(fixture);
}

TEST_CASE("GPU parity with a Gaussian beam kernel", "[gpu][parity]")
{
  auto fixture = make_fixture(true, 1);
  fixture.config.beam_size_enabled = true;
  fixture.item.beam_major_axis = xt::xarray<float>({1.F, 1.F});
  fixture.item.beam_minor_axis = xt::xarray<float>({1.F, 1.F});
  fixture.item.beam_position_angle = xt::xarray<float>({0.F, 0.F});
  run_parity(fixture);
}

TEST_CASE("GPU parity with mean subtraction, cutoffs and baseline flux", "[gpu][parity]")
{
  auto fixture = make_fixture(true, 1);
  fixture.config.subtract_mean = true;
  fixture.group.mean_image = xt::xtensor<float, 3>::from_shape({n_channels, n_y, n_x});
  fixture.group.mean_image->fill(0.5F);
  fixture.group.snr_cutoff = xt::xtensor<float, 3>::from_shape({n_channels, n_y, n_x});
  fixture.group.snr_cutoff->fill(0.5F);
  fixture.group.flux_cutoff = xt::xtensor<float, 3>::from_shape({n_channels, n_y, n_x});
  fixture.group.flux_cutoff->fill(0.5F);
  fixture.config.baseline_flux = xt::xtensor<float, 2>::from_shape({n_y, n_x});
  fixture.config.baseline_flux->fill(0.5F);
  run_parity(fixture);
}

TEST_CASE("GPU parity with histogram_detail 0", "[gpu][parity]")
{
  run_parity(make_fixture(true, 0));
}

TEST_CASE("GPU parity with boxsize 1", "[gpu][parity]")
{
  auto fixture = make_fixture(true, 1);
  fixture.config.boxsize = 1;
  run_parity(fixture);
}

TEST_CASE("GPU parity with boxsize 5", "[gpu][parity]")
{
  auto fixture = make_fixture(true, 1);
  fixture.config.boxsize = 5;
  run_parity(fixture);
}

TEST_CASE("GPU batch runner matches CPU batch runner", "[gpu][parity][batch]")
{
  auto fixture = make_fixture(true, 1);
  auto second_item = fixture.item;
  second_item.stokes_label = "Q";

  std::vector<PlaneBatchItem> cpu_items = {fixture.item, second_item};
  std::vector<PlaneBatchItem> gpu_items = {fixture.item, second_item};
  const std::vector<PlaneBatchGroup> groups = {fixture.group};

  const auto cpu_results = run_batch_core(cpu_items, groups, fixture.config, 2);
  const auto gpu_results = gpu::run_batch_core_gpu(gpu_items, groups, fixture.config);

  REQUIRE(cpu_results.size() == gpu_results.size());
  for (std::size_t i = 0; i < cpu_results.size(); ++i)
  {
    compare_plane_results(cpu_results[i], gpu_results[i]);
  }
}
