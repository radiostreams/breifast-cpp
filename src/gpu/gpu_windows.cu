#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "gpu_pipeline.hpp"

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

__global__ void compact_mask_kernel(const bool *const mask, const size_t n, size_t *const out_indices,
                                    size_t *const out_count, unsigned int *const scratch)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    *out_count = scratch[n - 1] + (mask[n - 1] ? 1U : 0U);
  }

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    if (mask[i])
    {
      out_indices[scratch[i]] = i;
    }
  }
}
} // namespace

void gpu_rms_noise_map(const float *plane, size_t n_y, size_t n_x, long boxsize, float *row_window_sums, float *out_rms,
                       GpuStream stream)
{
  if (n_y == 0 || n_x == 0)
  {
    throw std::runtime_error("gpu_rms_noise_map: n_y and n_x must be greater than 0");
  }

  // One thread per row for the horizontal pass, then one thread per column for the vertical pass.
  launch_per_element(window_rows_kernel, n_y, stream, plane, n_y, n_x, boxsize, row_window_sums);
  launch_per_element(window_cols_kernel, n_x, stream, row_window_sums, n_y, n_x, boxsize, out_rms);
}

void gpu_compact_mask(const bool *const mask, const size_t n, size_t *const out_indices, size_t *const out_count,
                      unsigned int *const scratch, GpuStream stream)
{
  if (n == 0)
  {
    throw std::runtime_error("gpu_compact_mask: zero pixels");
  }

  thrust::exclusive_scan(thrust::cuda::par.on(stream), mask, mask + n, scratch, 0U);

  constexpr size_t threads = 256;
  const dim3 block_size = {threads};
  const dim3 grid_size = {static_cast<unsigned int>((n + threads - 1) / threads)};
  compact_mask_kernel<<<grid_size, block_size, 0, stream>>>(mask, n, out_indices, out_count, scratch);
  CHECK_GPU(cudaGetLastError());
}
} // namespace breifast::gpu