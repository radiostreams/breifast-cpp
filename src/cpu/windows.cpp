#include "windows.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tracy/Tracy.hpp>

namespace breifast::plane_lot::detail
{

void rms_noise_map(std::span<const float> plane, const size_t n_y_size, const size_t n_x_size, const long boxsize,
                   std::span<float> row_window_sums, std::span<double> column_accumulators, std::span<float> out_rms)
{
  ZoneScopedN("compute_rms_noise_map");
  const size_t n_pixels = n_y_size * n_x_size;
  if (plane.size() != n_pixels)
  {
    throw std::runtime_error("rms_noise_map: plane size mismatch with n_y * n_x");
  }
  if (row_window_sums.size() != n_pixels)
  {
    throw std::runtime_error("rms_noise_map: row_window_sums size mismatch with n_y * n_x");
  }
  if (column_accumulators.size() != n_x_size)
  {
    throw std::runtime_error("rms_noise_map: column_accumulators size mismatch with n_x");
  }
  if (out_rms.size() != n_pixels)
  {
    throw std::runtime_error("rms_noise_map: out_rms size mismatch with n_y * n_x");
  }

  const long n_y = static_cast<long>(n_y_size);
  const long n_x = static_cast<long>(n_x_size);

  const long radius = boxsize / 2;
  const long window_size = std::max(boxsize, 1L);

  const auto min_x_of = [&](long i_x) { return std::max(i_x - radius, 0L); };
  const auto max_x_of = [&](long i_x) { return std::min(i_x + window_size - radius, n_x); };
  const auto min_y_of = [&](long i_y) { return std::max(i_y - radius, 0L); };
  const auto max_y_of = [&](long i_y) { return std::min(i_y + window_size - radius, n_y); };

  const float *const plane_ptr = plane.data();
  float *const row_sums = row_window_sums.data();
  double *const column_sums = column_accumulators.data();
  float *const out = out_rms.data();

  for (long i_y = 0; i_y < n_y; ++i_y)
  {
    double accumulator = 0.;
    for (long i_x = 0; i_x < max_x_of(0); ++i_x)
    {
      const double value = plane_ptr[i_y * n_x + i_x];
      accumulator += value * value;
    }

    for (long i_x = 0; i_x < n_x; ++i_x)
    {
      row_sums[i_y * n_x + i_x] = accumulator;

      const auto entering_column = i_x + window_size - radius;
      const auto leaving_column = i_x - radius;
      if (entering_column < n_x)
      {
        const double value = plane_ptr[i_y * n_x + entering_column];
        accumulator += value * value;
      }

      if (leaving_column >= 0)
      {
        const double value = plane_ptr[i_y * n_x + leaving_column];
        accumulator -= value * value;
      }
    }
  }

  std::fill(column_accumulators.begin(), column_accumulators.end(), 0.);

  for (long i_y = 0; i_y < max_y_of(0); ++i_y)
  {
    for (long i_x = 0; i_x < n_x; ++i_x)
    {
      column_sums[i_x] += row_sums[i_y * n_x + i_x];
    }
  }

  for (long i_y = 0; i_y < n_y; ++i_y)
  {
    const long rows_in_window = max_y_of(i_y) - min_y_of(i_y);
    for (long i_x = 0; i_x < n_x; ++i_x)
    {
      const long count = rows_in_window * (max_x_of(i_x) - min_x_of(i_x));
      out[i_y * n_x + i_x] = static_cast<float>(std::sqrt(column_sums[i_x] / static_cast<double>(count)));
    }

    const long entering_row = i_y + window_size - radius;
    const long leaving_row = i_y - radius;
    if (entering_row < n_y)
    {
      for (long i_x = 0; i_x < n_x; ++i_x)
      {
        column_sums[i_x] += row_sums[entering_row * n_x + i_x];
      }
    }
    if (leaving_row >= 0)
    {
      for (long i_x = 0; i_x < n_x; ++i_x)
      {
        column_sums[i_x] -= row_sums[leaving_row * n_x + i_x];
      }
    }
  }
}

} // namespace breifast::plane_lot::detail
