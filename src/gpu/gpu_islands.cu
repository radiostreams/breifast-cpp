#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "gpu_pipeline.hpp"

#include "gpu_device_utils.cuh"

namespace cg = cooperative_groups;

namespace breifast::gpu
{

namespace
{
constexpr unsigned int background = ~0U;

__global__ void init_labels_kernel(const bool *const mask, const size_t n, unsigned int *const labels)
{
  const size_t index = global_index();
  if (index < n)
  {
    labels[index] = mask[index] ? static_cast<unsigned int>(index) : background;
  }
}

__global__ void converge_labels_kernel(const bool *const mask, const size_t n_y, const size_t n_x,
                                       unsigned int *const labels, unsigned int *const changed)
{
  const cg::grid_group grid = cg::this_grid();
  const size_t n = n_y * n_x;

  while (true)
  {
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      *changed = 0U;
    }
    grid.sync();

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x)
    {
      if (!mask[index])
      {
        continue;
      }

      const size_t i_y = index / n_x;
      const size_t i_x = index % n_x;

      unsigned int best = labels[index];
      if (i_x > 0 && labels[index - 1] < best)
      {
        best = labels[index - 1];
      }
      if (i_x + 1 < n_x && labels[index + 1] < best)
      {
        best = labels[index + 1];
      }
      if (i_y > 0 && labels[index - n_x] < best)
      {
        best = labels[index - n_x];
      }
      if (i_y + 1 < n_y && labels[index + n_x] < best)
      {
        best = labels[index + n_x];
      }

      if (best < labels[index])
      {
        labels[index] = best;
        atomicOr(changed, 1U);
      }
    }
    grid.sync();

    // Every block must read `changed` before the next iteration resets it: a block that
    // read the reset value would leave the loop while the others call grid.sync() again.
    const unsigned int changed_now = *changed;
    grid.sync();

    if (changed_now == 0U)
    {
      break;
    }
  }
}

__global__ void mark_roots_kernel(const size_t n, const unsigned int *const labels, unsigned int *const root_ranks)
{
  const size_t index = global_index();
  if (index < n)
  {
    root_ranks[index] = labels[index] == static_cast<unsigned int>(index) ? 1U : 0U;
  }
}

__global__ void scatter_labels_kernel(const bool *const mask, const size_t n, unsigned int *const labels,
                                      const unsigned int *const root_ranks, unsigned int *const out_count)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    *out_count = root_ranks[n - 1];
  }
  const size_t i = global_index();
  if (i < n)
  {
    labels[i] = mask[i] ? root_ranks[labels[i]] : 0U;
  }
}

__global__ void island_boost_atomic_kernel(const unsigned int *const labels_per_pixel, const size_t n_pixels,
                                           const float *const baseline_flux, const unsigned int *const active_count,
                                           const unsigned int capacity, const float *const flux_per_label,
                                           unsigned long long *const keys)
{
  const size_t index = global_index();
  if (index >= n_pixels)
  {
    return;
  }
  const unsigned int label = labels_per_pixel[index];
  if (label == 0U || label > capacity || label > *active_count)
  {
    return;
  }

  const float ratio = flux_per_label[label - 1U] / baseline_flux[index];
  const float abs_value = fabsf(ratio);
  if (isnan(abs_value))
  {
    return;
  }

  const unsigned long long key = (static_cast<unsigned long long>(__float_as_uint(abs_value)) << 32) |
                                 static_cast<unsigned long long>(0xFFFFFFFFU - static_cast<unsigned int>(index));
  atomicMax(&keys[label], key);
}

__global__ void island_boost_finalize_kernel(const unsigned long long *const keys,
                                             const float *const baseline_flux, const unsigned int *const active_count,
                                             const unsigned int capacity, const float *const flux_per_label,
                                             float *const boost, bool *const found)
{
  const auto slot = global_index();
  if (slot >= capacity)
  {
    return;
  }

  const unsigned int label = static_cast<unsigned int>(slot) + 1U;
  if (label > *active_count)
  {
    boost[slot] = 0.F;
    found[slot] = false;
    return;
  }

  const unsigned long long key = keys[label];
  if (key == 0ULL)
  {
    boost[slot] = 0.F;
    found[slot] = false;
    return;
  }

  const unsigned int index = 0xFFFFFFFFU - static_cast<unsigned int>(key & 0xFFFFFFFFULL);
  boost[slot] = flux_per_label[slot] / baseline_flux[index];
  found[slot] = true;
}
} // namespace

void gpu_label_islands(const bool *const mask, const size_t n_y, const size_t n_x, unsigned int *const labels,
                       unsigned int *const out_count, unsigned int *const scratch, GpuStream stream)
{
  if (n_y == 0 || n_x == 0)
  {
    throw std::runtime_error("gpu_label_islands: n_y and n_x must be greater than 0");
  }

  const size_t n = n_y * n_x;
  launch_per_element(init_labels_kernel, n, stream, mask, n, labels);

  int device = 0;
  CHECK_GPU(cudaGetDevice(&device));
  int cooperative_launch = 0;
  CHECK_GPU(cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device));
  if (cooperative_launch == 0)
  {
    throw std::runtime_error("gpu_label_islands: device does not support cooperative launch");
  }

  int multiprocessor_count = 0;
  CHECK_GPU(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_multiprocessor = 0;
  CHECK_GPU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_multiprocessor, reinterpret_cast<const void *>(converge_labels_kernel),
      static_cast<int>(block_size), 0));
  const size_t max_cooperative_blocks =
      static_cast<size_t>(blocks_per_multiprocessor) * static_cast<size_t>(multiprocessor_count);
  if (max_cooperative_blocks == 0)
  {
    throw std::runtime_error("gpu_label_islands: no resident blocks available for cooperative launch");
  }

  size_t cooperative_blocks = (n + block_size - 1) / block_size;
  if (cooperative_blocks > max_cooperative_blocks)
  {
    cooperative_blocks = max_cooperative_blocks;
  }
  const dim3 cooperative_grid = {static_cast<unsigned int>(cooperative_blocks)};

  thrust::device_vector<unsigned int> changed(1);

  const bool *mask_arg = mask;
  size_t n_y_arg = n_y;
  size_t n_x_arg = n_x;
  unsigned int *labels_arg = labels;
  unsigned int *changed_arg = thrust::raw_pointer_cast(changed.data());
  void *converge_args[] = {&mask_arg, &n_y_arg, &n_x_arg, &labels_arg, &changed_arg};
  CHECK_GPU(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(converge_labels_kernel), cooperative_grid,
                                              dim3(block_size), converge_args, 0, stream));

  launch_per_element(mark_roots_kernel, n, stream, n, labels, scratch);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), scratch, scratch + n, scratch);
  launch_per_element(scatter_labels_kernel, n, stream, mask, n, labels, scratch, out_count);
}

void gpu_island_boost(const unsigned int *const labels_per_pixel, const float *const baseline_flux,
                      const size_t n_pixels, const unsigned int *const active_count, const unsigned int capacity,
                      const float *const flux_per_label, float *const boost, bool *const found, GpuStream stream)
{
  if (n_pixels == 0)
  {
    throw std::runtime_error("gpu_island_boost: n_pixels must be greater than 0");
  }
  if (active_count == nullptr)
  {
    throw std::runtime_error("gpu_island_boost: active_count must be non-null");
  }
  if (capacity == 0)
  {
    throw std::runtime_error("gpu_island_boost: capacity must be greater than 0");
  }

  thrust::device_vector<unsigned long long> keys(capacity + 1U, 0ULL);
  unsigned long long *keys_ptr = thrust::raw_pointer_cast(keys.data());

  launch_per_element(island_boost_atomic_kernel, n_pixels, stream, labels_per_pixel, n_pixels, baseline_flux,
                     active_count, capacity, flux_per_label, keys_ptr);
  launch_per_element(island_boost_finalize_kernel, capacity, stream, keys_ptr, baseline_flux, active_count, capacity,
                     flux_per_label, boost, found);
}

} // namespace breifast::gpu
