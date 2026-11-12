#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "gpu_plane_jobs.hpp"

using breifast::gpu::build_gaussian_beam_kernel_vector;
using breifast::gpu::GpuPlaneJobs;
using breifast::gpu::GpuPlaneJobSpec;
using Catch::Approx;

namespace
{
struct PlaneInputs
{
  static constexpr std::size_t n_y = 3;
  static constexpr std::size_t n_x = 4;
  static constexpr std::size_t n_pixels = n_y * n_x;
  static constexpr std::size_t n_channels = 2;

  std::vector<float> cube;
  std::vector<float> weights;
  std::vector<float> selfnoise_offset;
  std::vector<float> selfnoise_raw;
  std::vector<float> dde_noise;
  std::vector<std::uint8_t> accept_mask;
  std::vector<std::uint8_t> nonzero_channels;

  PlaneInputs()
  {
    cube.assign(n_pixels * n_channels, 1.F);
    weights.assign(n_channels, 1.F);
    selfnoise_offset.assign(n_pixels, 0.F);
    selfnoise_raw.assign(n_pixels, 1.F);
    dde_noise.assign(n_pixels, 0.F);
    accept_mask.assign(n_pixels * n_channels, 0U);
    accept_mask[0] = 1U;
    nonzero_channels.assign(n_channels, 1U);
  }

  GpuPlaneJobSpec spec() const
  {
    return GpuPlaneJobSpec{
        .cube = cube.data(),
        .weights = weights.data(),
        .nonzero_channels = reinterpret_cast<const bool *>(nonzero_channels.data()),
        .accept_mask = reinterpret_cast<const bool *>(accept_mask.data()),
        .selfnoise_offset = selfnoise_offset.data(),
        .selfnoise_raw = selfnoise_raw.data(),
        .dde_noise = dde_noise.data(),
        .stokes_label = "I",
        .bands = {{0, 2, 1.F, "band"}},
        .n_y = n_y,
        .n_x = n_x,
        .n_channels = n_channels,
        .boxsize = 1,
        .inner_boxsize = 0,
        .histogram_detail = 1,
        .max_sigma_bin = 5,
        .ddenoise_scaling_factor = 1.F,
        .reject_noise_threshold = 0.F,
        .threshold = 0.5F,
        .max_to_std_ratio_value = 1.F,
        .use_rms_noise_estimator = true,
        .subtract_mean = false,
    };
  }
};
} // namespace

TEST_CASE("build_plane_jobs creates one band input per band", "[gpu][plane_jobs]")
{
  PlaneInputs inputs;
  GpuPlaneJobs plane_jobs(inputs.spec());

  REQUIRE(plane_jobs.band_inputs.size() == 1);
  REQUIRE(plane_jobs.band_workspace_sizes.size() == 1);
  REQUIRE(plane_jobs.labels.size() == 1);
  CHECK(plane_jobs.labels[0].stokes_label == "I");
  CHECK(plane_jobs.labels[0].band_label == "band");
  CHECK(plane_jobs.band_workspace_sizes[0].n_pixels == PlaneInputs::n_pixels);
  CHECK(plane_jobs.band_workspace_sizes[0].stamp_pixels == 1);
  CHECK(plane_jobs.band_workspace_sizes[0].histogram_bins == 10);
  CHECK(plane_jobs.histogram_bins == 10);
  CHECK(plane_jobs.band_inputs[0].inv_band_size == Approx(0.5F));
}

TEST_CASE("build_plane_jobs uploads a fixed beam kernel", "[gpu][plane_jobs]")
{
  PlaneInputs inputs;
  auto spec = inputs.spec();
  const float kernel = 1.F;
  spec.beam_size_enabled = true;
  spec.fixed_beam_kernel = &kernel;
  spec.fixed_beam_halfsize = 0;

  GpuPlaneJobs plane_jobs(spec);
  REQUIRE(plane_jobs.band_inputs.size() == 1);
  CHECK(plane_jobs.band_inputs[0].beam_kernel != nullptr);
  CHECK(plane_jobs.band_inputs[0].beam_halfsize == 0);
  CHECK(plane_jobs.band_workspace_sizes[0].beam_stamp_pixels == 1);
  CHECK(plane_jobs.band_inputs[0].beam_kernel_sum == 1.F);
}

TEST_CASE("build_plane_jobs builds a per-band Gaussian beam kernel", "[gpu][plane_jobs]")
{
  PlaneInputs inputs;
  auto spec = inputs.spec();
  const std::vector<float> beam_major = {1.F, 1.F};
  const std::vector<float> beam_minor = {1.F, 1.F};
  const std::vector<float> beam_angle = {0.F, 0.F};
  spec.beam_size_enabled = true;
  spec.beam_major_axis = beam_major.data();
  spec.beam_minor_axis = beam_minor.data();
  spec.beam_position_angle = beam_angle.data();

  GpuPlaneJobs plane_jobs(spec);
  REQUIRE(plane_jobs.band_inputs.size() == 1);
  CHECK(plane_jobs.band_inputs[0].beam_kernel != nullptr);
  CHECK(plane_jobs.band_inputs[0].beam_halfsize == 4);
  CHECK(plane_jobs.band_workspace_sizes[0].beam_stamp_pixels == 81);
  // major = minor = 1 and angle 0 after weighting.
  const auto kernel = build_gaussian_beam_kernel_vector(4, 1.F, 1.F, 0.F);
  CHECK(plane_jobs.band_inputs[0].beam_kernel_sum == Approx(std::accumulate(kernel.begin(), kernel.end(), 0.F)));
}
