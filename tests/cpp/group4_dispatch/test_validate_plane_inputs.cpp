#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "plane_lot_dispatch.hpp"

using namespace breifast;

namespace
{
constexpr size_t n_channels = 2;
constexpr size_t n_y = 4;
constexpr size_t n_x = 5;

// A consistent batch: two planes of 2 x 4 x 5 sharing one group, one band over both channels. Each section breaks one
// thing; the shapes are all that matters, so the arrays are left uninitialised.
struct Batch
{
  std::vector<PlaneBatchItem> items;
  std::vector<PlaneBatchGroup> groups;
  PlaneConfig config;

  Batch()
  {
    PlaneBatchGroup group;
    group.accept_mask = xt::xtensor<bool, 3>::from_shape({n_channels, n_y, n_x});
    group.selfnoise_offset = xt::xtensor<float, 2>::from_shape({n_y, n_x});
    group.selfnoise_raw = xt::xtensor<float, 2>::from_shape({n_y, n_x});
    group.dde_noise = xt::xtensor<float, 2>::from_shape({n_y, n_x});
    groups.push_back(group);

    for (int i = 0; i < 2; ++i)
    {
      PlaneBatchItem item;
      item.stokes_cube = xt::xtensor<float, 3>::from_shape({n_channels, n_y, n_x});
      item.weights = xt::xtensor<float, 1>::from_shape({n_channels});
      item.nonzero_channels = xt::xtensor<bool, 1>::from_shape({n_channels});
      item.group_index = 0;
      items.push_back(item);
    }

    config.boxsize = 3;
    config.inner_boxsize = 0;
    config.max_sigma_bin = 20;
    config.band_slices = {BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B0~1"}};
  }
};

void check_rejects(const Batch &batch, const std::string &message)
{
  CHECK_THROWS_AS(validate_plane_inputs(batch.items, batch.groups, batch.config), std::invalid_argument);
  CHECK_THROWS_WITH(validate_plane_inputs(batch.items, batch.groups, batch.config), message);
}
} // namespace

TEST_CASE("validate_plane_inputs accepts a consistent batch", "[validate_plane_inputs]")
{
  Batch batch;
  CHECK_NOTHROW(validate_plane_inputs(batch.items, batch.groups, batch.config));

  // Optional inputs of the right shape, and a mean image that is not used.
  batch.groups[0].snr_cutoff = xt::xtensor<float, 3>::from_shape({n_channels, n_y, n_x});
  batch.groups[0].flux_cutoff = xt::xtensor<float, 3>::from_shape({n_channels, n_y, n_x});
  batch.groups[0].mean_image = xt::xtensor<float, 3>::from_shape({1, 1, 1});
  batch.config.baseline_flux = xt::xtensor<float, 2>::from_shape({n_y, n_x});
  CHECK_NOTHROW(validate_plane_inputs(batch.items, batch.groups, batch.config));
}

TEST_CASE("validate_plane_inputs rejects an invalid config", "[validate_plane_inputs]")
{
  Batch batch;
  SECTION("negative boxsize")
  {
    batch.config.boxsize = -1;
    check_rejects(batch, "config: boxsize and inner_boxsize must be non-negative");
  }
  SECTION("non-positive max_sigma_bin")
  {
    batch.config.max_sigma_bin = 0;
    check_rejects(batch, "config: max_sigma_bin must be positive");
  }
  SECTION("empty band")
  {
    batch.config.band_slices = {BandSlice{.start = 1, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B1"}};
    check_rejects(batch, "config: band B1 has the empty or negative channel range [1, 1)");
  }
  SECTION("fixed beam kernel of the wrong size")
  {
    batch.config.beam_size_enabled = true;
    batch.config.fixed_beam_halfsize = 1;
    batch.config.fixed_beam_kernel = xt::xtensor<double, 2>::from_shape({2, 2});
    check_rejects(batch, "config: fixed_beam_kernel has shape (2, 2), expected (3, 3)");
  }
}

TEST_CASE("validate_plane_inputs names the plane and the field of an inconsistent plane", "[validate_plane_inputs]")
{
  Batch batch;
  SECTION("group index out of range")
  {
    batch.items[1].group_index = 3;
    check_rejects(batch, "plane 1: group_index 3 is out of range for 1 groups");
  }
  SECTION("empty cube")
  {
    batch.items[0].stokes_cube = xt::xtensor<float, 3>::from_shape({n_channels, 0, n_x});
    check_rejects(batch, "plane 0: stokes_cube has the empty shape (2, 0, 5)");
  }
  SECTION("weights")
  {
    batch.items[1].weights = xt::xtensor<float, 1>::from_shape({1});
    check_rejects(batch, "plane 1: weights has shape (1), expected (2)");
  }
  SECTION("nonzero_channels")
  {
    batch.items[0].nonzero_channels = xt::xtensor<bool, 1>::from_shape({3});
    check_rejects(batch, "plane 0: nonzero_channels has shape (3), expected (2)");
  }
  SECTION("band past the channels")
  {
    batch.config.band_slices = {BandSlice{.start = 0, .stop = 3, .selfnoise_scale_factor = 1.F, .label = "B0~2"}};
    check_rejects(batch, "plane 0: band B0~2 stops at channel 3, past the 2 channels of stokes_cube");
  }
  SECTION("plane of another size than its group")
  {
    batch.items[1].stokes_cube = xt::xtensor<float, 3>::from_shape({n_channels, n_y, n_x + 1});
    check_rejects(batch, "plane 1: accept_mask has shape (2, 4, 5), expected (2, 4, 6)");
  }
  SECTION("missing Gaussian beam axes")
  {
    batch.config.beam_size_enabled = true;
    check_rejects(batch, "plane 0: beam_major_axis is missing");
  }
  SECTION("Gaussian beam axis of the wrong length")
  {
    batch.config.beam_size_enabled = true;
    for (auto &item : batch.items)
    {
      item.beam_major_axis = xt::xtensor<float, 1>::from_shape({n_channels});
      item.beam_minor_axis = xt::xtensor<float, 1>::from_shape({1});
      item.beam_position_angle = xt::xtensor<float, 1>::from_shape({n_channels});
    }
    check_rejects(batch, "plane 0: beam_minor_axis has shape (1), expected (2)");
  }
  SECTION("negative or non-finite Gaussian beam major axis")
  {
    batch.config.beam_size_enabled = true;
    for (auto &item : batch.items)
    {
      item.beam_major_axis = xt::xtensor<float, 1>({1.F, 1.F});
      item.beam_minor_axis = xt::xtensor<float, 1>({1.F, 1.F});
      item.beam_position_angle = xt::xtensor<float, 1>({0.F, 0.F});
    }
    (*batch.items[1].beam_major_axis)(0) = -1.F;
    check_rejects(batch, "plane 1: beam_major_axis has a negative or non-finite value");
    (*batch.items[1].beam_major_axis)(0) = std::numeric_limits<float>::quiet_NaN();
    check_rejects(batch, "plane 1: beam_major_axis has a negative or non-finite value");
  }
}

TEST_CASE("validate_plane_inputs names the group array of the wrong shape", "[validate_plane_inputs]")
{
  Batch batch;
  auto &group = batch.groups[0];
  SECTION("accept_mask")
  {
    group.accept_mask = xt::xtensor<bool, 3>::from_shape({n_channels, n_y, n_x + 1});
    check_rejects(batch, "plane 0: accept_mask has shape (2, 4, 6), expected (2, 4, 5)");
  }
  SECTION("selfnoise_offset")
  {
    group.selfnoise_offset = xt::xtensor<float, 2>::from_shape({3, 3});
    check_rejects(batch, "plane 0: selfnoise_offset has shape (3, 3), expected (4, 5)");
  }
  SECTION("selfnoise_raw")
  {
    group.selfnoise_raw = xt::xtensor<float, 2>::from_shape({n_x, n_y});
    check_rejects(batch, "plane 0: selfnoise_raw has shape (5, 4), expected (4, 5)");
  }
  SECTION("dde_noise")
  {
    group.dde_noise = xt::xtensor<float, 2>::from_shape({n_y, 1});
    check_rejects(batch, "plane 0: dde_noise has shape (4, 1), expected (4, 5)");
  }
  SECTION("mean_image when subtracted")
  {
    batch.config.subtract_mean = true;
    group.mean_image = xt::xtensor<float, 3>::from_shape({1, n_y, n_x});
    check_rejects(batch, "plane 0: mean_image has shape (1, 4, 5), expected (2, 4, 5)");
  }
  SECTION("snr_cutoff")
  {
    group.snr_cutoff = xt::xtensor<float, 3>::from_shape({n_channels, n_y, 1});
    check_rejects(batch, "plane 0: snr_cutoff has shape (2, 4, 1), expected (2, 4, 5)");
  }
  SECTION("flux_cutoff")
  {
    group.flux_cutoff = xt::xtensor<float, 3>::from_shape({1, n_y, n_x});
    check_rejects(batch, "plane 0: flux_cutoff has shape (1, 4, 5), expected (2, 4, 5)");
  }
  SECTION("baseline_flux")
  {
    batch.config.baseline_flux = xt::xtensor<float, 2>::from_shape({n_y, n_y});
    check_rejects(batch, "plane 0: baseline_flux has shape (4, 4), expected (4, 5)");
  }
}
