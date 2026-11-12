#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <random>
#include <sstream>
#include <vector>
#include <optional>

#include "gpu_plane.hpp"

using namespace breifast;
using Catch::Matchers::Equals;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

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

// A plane with structure, so that CPU/GPU differences show up. make_fixture() has a cube of ones with unit noise,
// where the SNR plane equals the input and many divergences are invisible. Here: seeded Gaussian noise in every
// channel, five Gaussian sources (sigma 1 pixel) with a spectral slope, non-uniform weights, an excluded border strip
// and noise maps that vary across the plane. Every band has several channels, so the CPU's single-channel SNR
// write-back does not happen (see the write-back tests below).
constexpr std::size_t structured_n_y = 64;
constexpr std::size_t structured_n_x = 80;
constexpr std::size_t structured_n_channels = 4;

Fixture make_structured_fixture(const bool use_rms)
{
  Fixture fixture;
  auto &cube = fixture.item.stokes_cube;
  cube.resize({structured_n_channels, structured_n_y, structured_n_x});
  std::mt19937 generator(12345);
  std::normal_distribution<float> noise(0.F, 1.F);
  for (std::size_t i = 0; i < cube.size(); ++i)
  {
    cube.data()[i] = noise(generator);
  }

  struct Source
  {
    float y;
    float x;
    float peak;
  };
  constexpr std::array<Source, 5> sources{
      {{12.F, 15.F, 20.F}, {30.F, 40.F, 15.F}, {50.F, 20.F, 25.F}, {20.F, 65.F, 12.F}, {45.F, 60.F, 18.F}}};
  for (std::size_t c = 0; c < structured_n_channels; ++c)
  {
    const float slope = 1.F + 0.1F * (static_cast<float>(c) - 1.5F);
    for (std::size_t y = 0; y < structured_n_y; ++y)
    {
      for (std::size_t x = 0; x < structured_n_x; ++x)
      {
        for (const auto &source : sources)
        {
          const float dy = static_cast<float>(y) - source.y;
          const float dx = static_cast<float>(x) - source.x;
          cube(c, y, x) += source.peak * slope * std::exp(-(dy * dy + dx * dx) / 2.F);
        }
      }
    }
  }
  fixture.item.weights = xt::xarray<float>({1.F, 0.8F, 1.2F, 0.6F});
  fixture.item.nonzero_channels = xt::xarray<bool>({true, true, true, true});
  fixture.item.stokes_label = "I";

  fixture.group.accept_mask.resize({structured_n_channels, structured_n_y, structured_n_x});
  fixture.group.accept_mask.fill(true);
  for (std::size_t c = 0; c < structured_n_channels; ++c)
  {
    for (std::size_t y = 0; y < structured_n_y; ++y)
    {
      for (std::size_t x = 0; x < 3; ++x)
      {
        fixture.group.accept_mask(c, y, x) = false;
      }
    }
  }
  fixture.group.selfnoise_offset.resize({structured_n_y, structured_n_x});
  fixture.group.selfnoise_raw.resize({structured_n_y, structured_n_x});
  fixture.group.dde_noise.resize({structured_n_y, structured_n_x});
  for (std::size_t y = 0; y < structured_n_y; ++y)
  {
    for (std::size_t x = 0; x < structured_n_x; ++x)
    {
      fixture.group.selfnoise_offset(y, x) = 0.01F * static_cast<float>(x) / static_cast<float>(structured_n_x);
      fixture.group.dde_noise(y, x) = 0.005F * static_cast<float>(y) / static_cast<float>(structured_n_y);
      fixture.group.selfnoise_raw(y, x) = 1.F + 0.1F * std::sin(0.3F * static_cast<float>(x));
    }
  }

  fixture.config.subtract_mean = false;
  fixture.config.use_rms_noise_estimator = use_rms;
  // Much larger than the sources, so that a source barely raises its own RMS noise.
  fixture.config.boxsize = 21;
  fixture.config.inner_boxsize = 2;
  fixture.config.histogram_detail = 2;
  fixture.config.max_sigma_bin = 20;
  fixture.config.threshold = 5.F;
  fixture.config.beam_size_enabled = false;
  fixture.config.band_slices = {
      BandSlice{.start = 0, .stop = 4, .selfnoise_scale_factor = 1.F, .label = "B0~3"},
      BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.5F, .label = "B0~1"},
      BandSlice{.start = 2, .stop = 4, .selfnoise_scale_factor = 1.5F, .label = "B2~3"},
  };
  return fixture;
}

// Bands are processed in reverse list order, so B1 and B0 run first and write their SNR planes back into channels 1
// and 0 on the CPU, and the wide band B0~3 then reads those channels.
std::vector<BandSlice> write_back_bands()
{
  return {
      BandSlice{.start = 0, .stop = 4, .selfnoise_scale_factor = 1.F, .label = "B0~3"},
      BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 4.F, .label = "B0"},
      BandSlice{.start = 1, .stop = 2, .selfnoise_scale_factor = 4.F, .label = "B1"},
  };
}

// A 256 x 256 single-channel plane with an isolated 100.0 spike every third pixel in both directions (85 x 85 = 7225
// single-pixel islands) on a deterministic background within +-0.3. In global-MAD mode the noise is about 0.25, so
// every spike is far above the threshold and no background pixel reaches it.
Fixture make_many_islands_fixture()
{
  constexpr std::size_t size = 256;
  Fixture fixture;
  auto &cube = fixture.item.stokes_cube;
  cube.resize({1, size, size});
  for (std::size_t y = 0; y < size; ++y)
  {
    for (std::size_t x = 0; x < size; ++x)
    {
      const bool spike = y % 3 == 1 && x % 3 == 1;
      cube(0, y, x) = spike ? 100.F : 0.3F * (static_cast<float>((y * 7 + x * 13) % 11) - 5.F) / 5.F;
    }
  }
  fixture.item.weights = xt::xarray<float>({1.F});
  fixture.item.nonzero_channels = xt::xarray<bool>({true});
  fixture.item.stokes_label = "I";

  fixture.group.accept_mask.resize({1, size, size});
  fixture.group.accept_mask.fill(true);
  fixture.group.selfnoise_offset.resize({size, size});
  fixture.group.selfnoise_offset.fill(0.F);
  fixture.group.selfnoise_raw.resize({size, size});
  fixture.group.selfnoise_raw.fill(1.F);
  fixture.group.dde_noise.resize({size, size});
  fixture.group.dde_noise.fill(0.F);

  fixture.config.subtract_mean = false;
  fixture.config.use_rms_noise_estimator = false;
  fixture.config.boxsize = 5;
  fixture.config.inner_boxsize = 0;
  fixture.config.histogram_detail = 0;
  fixture.config.max_sigma_bin = 5;
  fixture.config.threshold = 5.F;
  fixture.config.beam_size_enabled = false;
  fixture.config.band_slices = {BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"}};
  return fixture;
}

void check_close(const double actual, const double expected, const char *name)
{
  INFO(name);
  if (std::isnan(actual) && std::isnan(expected))
  {
    return;
  }
  // Absolute for values near 0, relative for large ones: the structured fixture has fluxes and moments far above 1,
  // where float rounding (operation order, FMA contraction) differs between the host compiler and nvcc.
  CHECK_THAT(actual, WithinAbs(expected, 1e-5) || WithinRel(expected, 1e-5));
}

/// The tolerance of check_close() as a predicate, for whole arrays.
bool values_close(const double actual, const double expected)
{
  if (std::isnan(actual) && std::isnan(expected))
  {
    return true;
  }
  const double difference = std::abs(actual - expected);
  return difference <= 1e-5 || difference <= 1e-5 * std::max(std::abs(actual), std::abs(expected));
}

/// One check for the first `n` elements of an array, reporting how many differ and the first difference, instead of
/// one check per element: a mismatching variance-stats map would otherwise log every pixel.
template <typename Values, typename Equal>
void check_all(const Values &gpu, const Values &cpu, const std::size_t n, const char *name, const Equal &equal)
{
  std::size_t mismatches = 0;
  std::size_t first = 0;
  for (std::size_t i = 0; i < n; ++i)
  {
    if (!equal(gpu[i], cpu[i]))
    {
      if (mismatches == 0)
      {
        first = i;
      }
      ++mismatches;
    }
  }
  std::ostringstream message;
  message << name << ": " << mismatches << " of " << n << " differ";
  if (mismatches > 0)
  {
    message << std::setprecision(10) << ", first at index " << first << ": gpu " << gpu[first] << ", cpu "
            << cpu[first];
  }
  INFO(message.str());
  CHECK(mismatches == 0);
}

void compare_plane_results(const ProcessPlaneResult &cpu, const ProcessPlaneResult &gpu)
{
  const auto exact = [](const auto &a, const auto &b) { return a == b; };
  const auto approx = [](const double a, const double b) { return values_close(a, b); };

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
  check_all(gpu.median_noises, cpu.median_noises, cpu.median_noises.size(), "median_noise", approx);

  const auto &cpu_detections = cpu.detections;
  const auto &gpu_detections = gpu.detections;
  REQUIRE(cpu_detections.size() == gpu_detections.size());
  const std::size_t n = cpu_detections.size();
  check_all(gpu_detections.stokes_label(), cpu_detections.stokes_label(), n, "stokes_label", exact);
  check_all(gpu_detections.band_label(), cpu_detections.band_label(), n, "band_label", exact);
  check_all(gpu_detections.peak_y(), cpu_detections.peak_y(), n, "peak_y", exact);
  check_all(gpu_detections.peak_x(), cpu_detections.peak_x(), n, "peak_x", exact);
  check_all(gpu_detections.island_size(), cpu_detections.island_size(), n, "island_size", exact);
  check_all(gpu_detections.flux(), cpu_detections.flux(), n, "flux", approx);
  check_all(gpu_detections.local_noise(), cpu_detections.local_noise(), n, "local_noise", approx);
  check_all(gpu_detections.local_noise_median_ratio(), cpu_detections.local_noise_median_ratio(), n,
            "local_noise_median_ratio", approx);
  check_all(gpu_detections.signal_to_noise(), cpu_detections.signal_to_noise(), n, "signal_to_noise", approx);
  check_all(gpu_detections.signal_to_noise_rms(), cpu_detections.signal_to_noise_rms(), n, "signal_to_noise_rms",
            approx);
  check_all(gpu_detections.signal_to_noise_extended_box(), cpu_detections.signal_to_noise_extended_box(), n,
            "signal_to_noise_extended_box", approx);
  check_all(gpu_detections.signal_to_noise_local_mad(), cpu_detections.signal_to_noise_local_mad(), n,
            "signal_to_noise_local_mad", approx);
  check_all(gpu_detections.signal_to_noise_global_mad(), cpu_detections.signal_to_noise_global_mad(), n,
            "signal_to_noise_global_mad", approx);
  check_all(gpu_detections.self_noise(), cpu_detections.self_noise(), n, "self_noise", approx);
  check_all(gpu_detections.excess_noise_factor(), cpu_detections.excess_noise_factor(), n, "excess_noise_factor",
            approx);
  check_all(gpu_detections.beam_chi2(), cpu_detections.beam_chi2(), n, "beam_chi2", approx);
  check_all(gpu_detections.boost(), cpu_detections.boost(), n, "boost", approx);

  REQUIRE(cpu.variance_stats.has_value() == gpu.variance_stats.has_value());
  if (cpu.variance_stats.has_value())
  {
    const auto &cpu_variance = *cpu.variance_stats;
    const auto &gpu_variance = *gpu.variance_stats;
    REQUIRE(cpu_variance.sum_pow2.shape() == gpu_variance.sum_pow2.shape());
    const std::size_t n_pixels = cpu_variance.sum_pow2.size();
    check_all(gpu_variance.sum_pow2.data(), cpu_variance.sum_pow2.data(), n_pixels, "sum_pow2", approx);
    check_all(gpu_variance.sum_pow3.data(), cpu_variance.sum_pow3.data(), n_pixels, "sum_pow3", approx);
    check_all(gpu_variance.sum_pow4.data(), cpu_variance.sum_pow4.data(), n_pixels, "sum_pow4", approx);
    REQUIRE(cpu_variance.nonzero_channels_mask.shape() == gpu_variance.nonzero_channels_mask.shape());
    check_all(gpu_variance.nonzero_channels_mask.data(), cpu_variance.nonzero_channels_mask.data(),
              cpu_variance.nonzero_channels_mask.size(), "nonzero_channels_mask", exact);
  }
}
/// Runs the CPU and GPU on copies of the fixture's item, compares the results and returns the CPU one.
ProcessPlaneResult run_parity(Fixture fixture)
{
  auto cpu_item = fixture.item;
  auto gpu_item = fixture.item;
  ProcessBandScratch scratch;
  const auto cpu_result = process_plane(scratch, fixture.config, cpu_item, fixture.group);
  const auto gpu_result = gpu::gpu_process_plane(gpu_item, fixture.group, fixture.config);
  compare_plane_results(cpu_result, gpu_result);
  return cpu_result;
}
} // namespace

TEST_CASE("GPU plane result matches CPU in RMS noise mode", "[gpu][parity]")
{
  auto fixture = make_fixture(true, 1);
  auto cpu_item = fixture.item;
  auto gpu_item = fixture.item;
  ProcessBandScratch scratch;
  const auto cpu_result = process_plane(scratch, fixture.config, cpu_item, fixture.group);
  const auto gpu_result = gpu::gpu_process_plane(gpu_item, fixture.group, fixture.config);
  compare_plane_results(cpu_result, gpu_result);
}

TEST_CASE("GPU plane result matches CPU in global-MAD noise mode", "[gpu][parity]")
{
  auto fixture = make_fixture(false, 2);
  auto cpu_item = fixture.item;
  auto gpu_item = fixture.item;
  ProcessBandScratch scratch;
  const auto cpu_result = process_plane(scratch, fixture.config, cpu_item, fixture.group);
  const auto gpu_result = gpu::gpu_process_plane(gpu_item, fixture.group, fixture.config);
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

TEST_CASE("GPU parity on a structured plane in RMS noise mode", "[gpu][parity]")
{
  const auto cpu_result = run_parity(make_structured_fixture(true));
  INFO("the fixture must produce detections to be meaningful");
  CHECK(cpu_result.detections.size() >= 5);
}

TEST_CASE("GPU parity on a structured plane in global-MAD noise mode", "[gpu][parity]")
{
  const auto cpu_result = run_parity(make_structured_fixture(false));
  INFO("the fixture must produce detections to be meaningful");
  CHECK(cpu_result.detections.size() >= 5);
}

// Control for the next test: the CPU really does write the SNR of single-channel bands back into the cube with
// write_back_bands(), so a failure there is the GPU missing it, not a fixture that never triggers it.
TEST_CASE("CPU writes the SNR of single-channel bands back into the cube", "[parity][fixture]")
{
  auto fixture = make_structured_fixture(true);
  fixture.config.band_slices = write_back_bands();
  auto item = fixture.item;
  ProcessBandScratch scratch;
  process_plane(scratch, fixture.config, item, fixture.group);

  const auto &before = fixture.item.stokes_cube;
  const auto &after = item.stokes_cube;
  CHECK(after(0, 30, 40) != before(0, 30, 40));
  CHECK(after(1, 30, 40) != before(1, 30, 40));
  CHECK(after(2, 30, 40) == before(2, 30, 40));
  CHECK(after(3, 30, 40) == before(3, 30, 40));
}

// Known GPU bug: the GPU neither writes the SNR back nor computes the spectral variance stats after the bands, so the
// wide band's detections and the variance stats differ from the CPU. Remove [!shouldfail] when the refactor fixes it
// (gpu-refactor-plan.md, step 15).
TEST_CASE("GPU parity when a wide band follows single-channel bands", "[gpu][parity][!shouldfail]")
{
  auto fixture = make_structured_fixture(true);
  fixture.config.band_slices = write_back_bands();
  run_parity(fixture);
}

// Control for the next test: the fixture has more islands than the GPU's former 4096 cap.
TEST_CASE("CPU finds more than 4096 islands in the many-islands fixture", "[parity][fixture]")
{
  auto fixture = make_many_islands_fixture();
  ProcessBandScratch scratch;
  const auto result = process_plane(scratch, fixture.config, fixture.item, fixture.group);
  CHECK(result.detections.size() == 85 * 85);
}

// The GPU used to cap a band at 4096 islands; it now sizes its island buffers from the count, as the CPU has no limit.
TEST_CASE("GPU parity with more than 4096 islands", "[gpu][parity]")
{
  run_parity(make_many_islands_fixture());
}
