#include "island_stamps.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/views/xview.hpp>

namespace breifast::plane_lot::detail
{

void stamp_extract(std::span<const float> plane, size_t n_y, size_t n_x, long centre_y, long centre_x, long radius,
                   std::span<float> out_stamp)
{
  if (plane.size() != n_y * n_x)
  {
    throw std::runtime_error("stamp_extract: plane size mismatch with n_y * n_x");
  }
  if (radius < 0)
  {
    throw std::runtime_error("stamp_extract: negative radius");
  }
  const long size = 2 * radius + 1;
  if (out_stamp.size() != static_cast<size_t>(size) * static_cast<size_t>(size))
  {
    throw std::runtime_error("stamp_extract: out_stamp size mismatch with (2 * radius + 1)^2");
  }

  std::fill(out_stamp.begin(), out_stamp.end(), 0.F);

  // Stamp row r and column c hold plane(first_y + r, first_x + c) when that pixel exists.
  const long first_y = centre_y - radius;
  const long first_x = centre_x - radius;
  const long row_begin = std::max(0L, -first_y);
  const long row_end = std::min(size, static_cast<long>(n_y) - first_y);
  const long column_begin = std::max(0L, -first_x);
  const long column_end = std::min(size, static_cast<long>(n_x) - first_x);
  if (column_begin >= column_end)
  {
    return;
  }
  for (long row = row_begin; row < row_end; ++row)
  {
    const float *const source = plane.data() + (first_y + row) * static_cast<long>(n_x) + first_x + column_begin;
    std::copy_n(source, column_end - column_begin, out_stamp.data() + row * size + column_begin);
  }
}

void stamp_zero_inner_box(std::span<float> stamp, long radius, long inner_boxsize)
{
  if (radius < 0)
  {
    throw std::runtime_error("stamp_zero_inner_box: negative radius");
  }
  if (inner_boxsize < 0)
  {
    throw std::runtime_error("stamp_zero_inner_box: negative inner_boxsize");
  }
  const long size = 2 * radius + 1;
  if (stamp.size() != static_cast<size_t>(size) * static_cast<size_t>(size))
  {
    throw std::runtime_error("stamp_zero_inner_box: stamp size mismatch with (2 * radius + 1)^2");
  }

  const long half = std::min(inner_boxsize, radius);
  for (long row = radius - half; row <= radius + half; ++row)
  {
    float *const first = stamp.data() + row * size + (radius - half);
    std::fill(first, first + (2 * half + 1), 0.F);
  }
}

float stamp_abs_extreme(std::span<const float> stamp)
{
  float lowest = std::numeric_limits<float>::max();
  float highest = std::numeric_limits<float>::lowest();
  for (const float value : stamp)
  {
    lowest = std::min(lowest, value);
    highest = std::max(highest, value);
  }
  return std::max(std::abs(lowest), std::abs(highest));
}

DonutStats stamp_donut_stats(std::span<const float> stamp, long radius)
{
  if (radius < 0)
  {
    throw std::runtime_error("stamp_donut_stats: negative radius");
  }
  const size_t size = static_cast<size_t>(2 * radius + 1);
  if (stamp.size() != size * size)
  {
    throw std::runtime_error("stamp_donut_stats: stamp size mismatch with (2 * radius + 1)^2");
  }

  const auto sub_image = xt::adapt(stamp.data(), stamp.size(), xt::no_ownership(), std::array<size_t, 2>{size, size});
  const auto donut = xt::filter(sub_image, xt::not_equal(sub_image, 0.F));
  const auto rms = donut.size() == 0 ? 0.F : xt::stddev(donut)();
  const auto lmad = donut.size() == 0 ? 0.F : xt::median(xt::abs(donut)) * 1.4826F;
  return {rms, lmad};
}

float stamp_beam_chi2(std::span<const float> beam_stamp, std::span<const float> beam_kernel, long halfsize,
                      float adjacent_noise)
{
  if (halfsize < 0)
  {
    throw std::runtime_error("stamp_beam_chi2: negative halfsize");
  }
  const size_t size = static_cast<size_t>(2 * halfsize + 1);
  if (beam_stamp.size() != size * size)
  {
    throw std::runtime_error("stamp_beam_chi2: beam_stamp size mismatch with (2 * halfsize + 1)^2");
  }
  if (beam_kernel.size() != size * size)
  {
    throw std::runtime_error("stamp_beam_chi2: beam_kernel size mismatch with (2 * halfsize + 1)^2");
  }

  const auto beam_sub_image =
      xt::adapt(beam_stamp.data(), beam_stamp.size(), xt::no_ownership(), std::array<size_t, 2>{size, size});
  const auto kernel =
      xt::adapt(beam_kernel.data(), beam_kernel.size(), xt::no_ownership(), std::array<size_t, 2>{size, size});

  float beam_chi2_value = 0.F;
  const auto fit = kernel * (xt::sum(beam_sub_image)() / xt::sum(kernel)());
  const auto mask = xt::not_equal(beam_sub_image, 0.F);
  const auto diff = xt::filter(beam_sub_image, mask) - xt::filter(fit, mask);
  const auto mask_count = xt::sum(mask)();

  if (mask_count > 0 && adjacent_noise != 0.F)
  {
    beam_chi2_value = xt::sum(diff * diff)() / (adjacent_noise * adjacent_noise * mask_count);
  }
  return beam_chi2_value;
}

} // namespace breifast::plane_lot::detail
