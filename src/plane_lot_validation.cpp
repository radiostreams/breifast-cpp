#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "plane_lot_dispatch.hpp"

namespace breifast
{
namespace
{
template <typename Shape> std::string format_shape(const Shape &shape)
{
  std::ostringstream out;
  out << '(';
  for (std::size_t i = 0; i < shape.size(); ++i)
  {
    out << (i > 0 ? ", " : "") << shape[i];
  }
  out << ')';
  return out.str();
}

[[noreturn]] void fail(const std::string &message)
{
  throw std::invalid_argument(message);
}

template <typename Array, std::size_t N>
void require_shape(const std::string &where, const char *name, const Array &array,
                   const std::array<std::size_t, N> &expected)
{
  const auto &shape = array.shape();
  if (shape.size() != N || !std::equal(expected.begin(), expected.end(), shape.begin()))
  {
    fail(where + name + " has shape " + format_shape(shape) + ", expected " + format_shape(expected));
  }
}

template <typename Array, std::size_t N>
void require_present(const std::string &where, const char *name, const std::optional<Array> &array,
                     const std::array<std::size_t, N> &expected)
{
  if (!array.has_value())
  {
    fail(where + name + " is missing");
  }
  require_shape(where, name, *array, expected);
}

void validate_config(const PlaneConfig &config)
{
  if (config.boxsize < 0 || config.inner_boxsize < 0)
  {
    fail("config: boxsize and inner_boxsize must be non-negative");
  }
  if (config.max_sigma_bin <= 0)
  {
    fail("config: max_sigma_bin must be positive");
  }
  for (const auto &band : config.band_slices)
  {
    if (band.start < 0 || band.stop <= band.start)
    {
      fail("config: band " + band.label + " has the empty or negative channel range [" + std::to_string(band.start) +
           ", " + std::to_string(band.stop) + ")");
    }
  }
  if (config.beam_size_enabled && config.fixed_beam_kernel.has_value())
  {
    if (config.fixed_beam_halfsize < 0)
    {
      fail("config: fixed_beam_halfsize must be non-negative");
    }
    const auto side = static_cast<std::size_t>(2 * config.fixed_beam_halfsize + 1);
    require_shape("config: ", "fixed_beam_kernel", *config.fixed_beam_kernel, std::array{side, side});
  }
}
} // namespace

void validate_plane_inputs(const std::vector<PlaneBatchItem> &items, const std::vector<PlaneBatchGroup> &groups,
                           const PlaneConfig &config)
{
  validate_config(config);
  const bool gaussian_beam = config.beam_size_enabled && !config.fixed_beam_kernel.has_value();

  for (std::size_t i = 0; i < items.size(); ++i)
  {
    const auto &item = items[i];
    const std::string where = "plane " + std::to_string(i) + ": ";
    if (item.group_index >= groups.size())
    {
      fail(where + "group_index " + std::to_string(item.group_index) + " is out of range for " +
           std::to_string(groups.size()) + " groups");
    }
    const auto &group = groups[item.group_index];

    const auto [n_channels, n_y, n_x] = item.stokes_cube.shape();
    if (n_channels == 0 || n_y == 0 || n_x == 0)
    {
      fail(where + "stokes_cube has the empty shape " + format_shape(item.stokes_cube.shape()));
    }
    const std::array cube_shape{n_channels, n_y, n_x};
    const std::array plane_shape{n_y, n_x};
    const std::array channel_shape{n_channels};

    require_shape(where, "weights", item.weights, channel_shape);
    require_shape(where, "nonzero_channels", item.nonzero_channels, channel_shape);
    for (const auto &band : config.band_slices)
    {
      if (static_cast<std::size_t>(band.stop) > n_channels)
      {
        fail(where + "band " + band.label + " stops at channel " + std::to_string(band.stop) + ", past the " +
             std::to_string(n_channels) + " channels of stokes_cube");
      }
    }
    if (gaussian_beam)
    {
      require_present(where, "beam_major_axis", item.beam_major_axis, channel_shape);
      require_present(where, "beam_minor_axis", item.beam_minor_axis, channel_shape);
      require_present(where, "beam_position_angle", item.beam_position_angle, channel_shape);
      // A band's beam half size is int(3 * mean(beam_major_axis)) + 1: undefined for a non-finite value, and negative,
      // so meaningless as a size, for negative ones.
      const auto &major = *item.beam_major_axis;
      if (!std::all_of(major.cbegin(), major.cend(),
                       [](const float value) { return std::isfinite(value) && value >= 0.F; }))
      {
        fail(where + "beam_major_axis has a negative or non-finite value");
      }
    }

    require_shape(where, "accept_mask", group.accept_mask, cube_shape);
    require_shape(where, "selfnoise_offset", group.selfnoise_offset, plane_shape);
    require_shape(where, "selfnoise_raw", group.selfnoise_raw, plane_shape);
    require_shape(where, "dde_noise", group.dde_noise, plane_shape);
    if (config.subtract_mean && group.mean_image.has_value())
    {
      require_shape(where, "mean_image", *group.mean_image, cube_shape);
    }
    if (group.snr_cutoff.has_value())
    {
      require_shape(where, "snr_cutoff", *group.snr_cutoff, cube_shape);
    }
    if (group.flux_cutoff.has_value())
    {
      require_shape(where, "flux_cutoff", *group.flux_cutoff, cube_shape);
    }
    if (config.baseline_flux.has_value())
    {
      require_shape(where, "baseline_flux", *config.baseline_flux, plane_shape);
    }
  }
}
} // namespace breifast
