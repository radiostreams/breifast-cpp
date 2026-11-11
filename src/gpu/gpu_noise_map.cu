#include <cassert>

#include "gpu_kernels.hpp"

#include "gpu_device_utils.cuh"

namespace breifast::gpu
{

namespace
{
__device__ inline long min_of(long i, long radius)
{
  const long bound = i - radius;
  return bound > 0 ? bound : 0;
}

__device__ inline long max_of(long i, long window_size, long radius, long n)
{
  const long bound = i + window_size - radius;
  return bound < n ? bound : n;
}

__global__ void window_rows_kernel(const float *plane, const size_t n_y, const size_t n_x, long box_size,
                                   float *row_window_sums)
{
  const auto i_y = global_index();
  const auto radius = box_size / 2;
  const auto window_size = box_size > 0 ? box_size : 1;
  if (i_y < n_y)
  {
    double acc = 0.;
    for (long i_x = 0; i_x < max_of(0, window_size, radius, static_cast<long>(n_x)); ++i_x)
    {
      const double value = plane[i_y * n_x + i_x];
      acc += value * value;
    }

    for (long i_x = 0; i_x < n_x; ++i_x)
    {
      row_window_sums[i_y * n_x + i_x] = acc;
      const auto entering_column = i_x + window_size - radius;
      const auto leaving_column = i_x - radius;
      if (entering_column < n_x)
      {
        const double value = plane[i_y * n_x + entering_column];
        acc += value * value;
      }

      if (leaving_column >= 0)
      {
        const double value = plane[i_y * n_x + leaving_column];
        acc -= value * value;
      }
    }
  }
}

__global__ void window_cols_kernel(const float *row_window_sums, const size_t n_y, const size_t n_x, long box_size,
                                   float *out)
{
  const auto i_x = global_index();
  const auto radius = box_size / 2;
  const auto window_size = box_size > 0 ? box_size : 1;
  if (i_x < n_x)
  {
    double acc = 0.;

    for (long i_y = 0; i_y < max_of(0, window_size, radius, static_cast<long>(n_y)); ++i_y)
    {
      acc += row_window_sums[i_y * n_x + i_x];
    }

    for (long i_y = 0; i_y < n_y; ++i_y)
    {
      const long dx = max_of(i_x, window_size, radius, static_cast<long>(n_x)) - min_of(i_x, radius);
      const long dy = max_of(i_y, window_size, radius, static_cast<long>(n_y)) - min_of(i_y, radius);
      const long count = dx * dy;

      out[i_y * n_x + i_x] = static_cast<float>(sqrt(acc / static_cast<double>(count)));

      const long entering_row = i_y + window_size - radius;
      const long leaving_row = i_y - radius;
      if (entering_row < n_y)
      {
        acc += row_window_sums[entering_row * n_x + i_x];
      }
      if (leaving_row >= 0)
      {
        acc -= row_window_sums[leaving_row * n_x + i_x];
      }
    }
  }
}

// The base noise is max(rms[p], *base_noise) with an RMS map, else *base_noise.
__global__ void combine_noise_map_kernel(const float *rms, const float *base_noise, const float *selfnoise_offset,
                                         const float *dde_noise, const float selfnoise_scale, const float dde_scale,
                                         float *out_noise_map, const size_t n_pixels)
{
  const auto i_pixel = global_index();
  if (i_pixel < n_pixels)
  {
    const float base = rms != nullptr ? fmaxf(rms[i_pixel], *base_noise) : *base_noise;
    out_noise_map[i_pixel] =
        sqrtf(base * base + selfnoise_scale * selfnoise_offset[i_pixel] + dde_scale * dde_noise[i_pixel]);
  }
}
} // namespace

void gpu_rms_noise_map(const float *plane, size_t n_y, size_t n_x, long boxsize, float *row_window_sums, float *out_rms,
                       GpuStream stream)
{
  assert(n_y > 0 && n_x > 0);

  // One thread per row for the horizontal pass, then one thread per column for the vertical pass.
  launch_per_element(window_rows_kernel, n_y, stream, plane, n_y, n_x, boxsize, row_window_sums);
  launch_per_element(window_cols_kernel, n_x, stream, row_window_sums, n_y, n_x, boxsize, out_rms);
}

void gpu_combine_noise_map(const float *rms, const float *base_noise, const float *selfnoise_offset,
                           const float *dde_noise, const float selfnoise_scale, const float dde_scale,
                           float *out_noise_map, const size_t n_pixels, GpuStream stream)
{
  assert(n_pixels > 0);
  launch_per_element(combine_noise_map_kernel, n_pixels, stream, rms, base_noise, selfnoise_offset, dde_noise,
                     selfnoise_scale, dde_scale, out_noise_map, n_pixels);
}
} // namespace breifast::gpu