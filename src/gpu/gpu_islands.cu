#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "gpu_pipeline.hpp"

#include "gpu_runtime.hpp"

namespace cg = cooperative_groups;

namespace breifast::gpu
{

namespace
{
constexpr size_t block_dim = 256;
constexpr unsigned int background = ~0U;

__global__ void init_labels_kernel(const bool *const mask, const size_t n, unsigned int *const labels)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
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

    if (*changed == 0U)
    {
      break;
    }
  }
}

__global__ void mark_roots_kernel(const size_t n, const unsigned int *const labels, unsigned int *const root_ranks)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
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

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    labels[i] = mask[i] ? root_ranks[labels[i]] : 0U;
  }
}

constexpr size_t boost_block_size = 256;
constexpr size_t max_boost_blocks = 4096;

__global__ void island_boost_atomic_kernel(const unsigned int *const labels_per_pixel, const size_t n_pixels,
                                           const float *const baseline_flux, const unsigned int *const active_count,
                                           const unsigned int capacity, const float *const flux_per_label,
                                           unsigned long long *const keys)
{
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < n_pixels;
       index += static_cast<size_t>(blockDim.x) * gridDim.x)
  {
    const unsigned int label = labels_per_pixel[index];
    if (label == 0U || label > capacity || label > *active_count)
    {
      continue;
    }

    const float ratio = flux_per_label[label - 1U] / baseline_flux[index];
    const float abs_value = fabsf(ratio);
    if (isnan(abs_value))
    {
      continue;
    }

    const unsigned long long key = (static_cast<unsigned long long>(__float_as_uint(abs_value)) << 32) |
                                   static_cast<unsigned long long>(0xFFFFFFFFU - static_cast<unsigned int>(index));
    atomicMax(&keys[label], key);
  }
}

__global__ void island_boost_finalize_kernel(const unsigned long long *const keys,
                                             const float *const baseline_flux, const unsigned int *const active_count,
                                             const unsigned int capacity, const float *const flux_per_label,
                                             float *const boost, bool *const found)
{
  const unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= capacity)
  {
    return;
  }

  const unsigned int label = slot + 1U;
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
  const dim3 block_size = {block_dim};
  const dim3 grid_size = {static_cast<unsigned int>((n + block_dim - 1) / block_dim)};

  init_labels_kernel<<<grid_size, block_size, 0, stream>>>(mask, n, labels);
  CHECK_LAST_GPU_ERROR();

  int device = 0;
  CHECK_GPU_ERROR(cudaGetDevice(&device));
  int cooperative_launch = 0;
  CHECK_GPU_ERROR(cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device));
  if (cooperative_launch == 0)
  {
    throw std::runtime_error("gpu_label_islands: device does not support cooperative launch");
  }

  int multiprocessor_count = 0;
  CHECK_GPU_ERROR(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_multiprocessor = 0;
  CHECK_GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_multiprocessor, reinterpret_cast<const void *>(converge_labels_kernel),
      static_cast<int>(block_dim), 0));
  const size_t max_cooperative_blocks =
      static_cast<size_t>(blocks_per_multiprocessor) * static_cast<size_t>(multiprocessor_count);
  if (max_cooperative_blocks == 0)
  {
    throw std::runtime_error("gpu_label_islands: no resident blocks available for cooperative launch");
  }

  size_t cooperative_blocks = (n + block_dim - 1) / block_dim;
  if (cooperative_blocks > max_cooperative_blocks)
  {
    cooperative_blocks = max_cooperative_blocks;
  }
  const dim3 cooperative_grid = {static_cast<unsigned int>(cooperative_blocks)};

  unsigned int *changed = nullptr;
  gpuMallocAsync(&changed, sizeof(unsigned int), stream);

  const bool *mask_arg = mask;
  size_t n_y_arg = n_y;
  size_t n_x_arg = n_x;
  unsigned int *labels_arg = labels;
  unsigned int *changed_arg = changed;
  void *converge_args[] = {&mask_arg, &n_y_arg, &n_x_arg, &labels_arg, &changed_arg};
  CHECK_GPU_ERROR(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(converge_labels_kernel), cooperative_grid,
                                              block_size, converge_args, 0, stream));

  gpuFreeAsync(changed, stream);

  mark_roots_kernel<<<grid_size, block_size, 0, stream>>>(n, labels, scratch);
  CHECK_LAST_GPU_ERROR();
  thrust::inclusive_scan(thrust::cuda::par.on(stream), scratch, scratch + n, scratch);

  scatter_labels_kernel<<<grid_size, block_size, 0, stream>>>(mask, n, labels, scratch, out_count);
  CHECK_LAST_GPU_ERROR();
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

  unsigned long long *keys = nullptr;
  gpuMallocAsync(&keys, (capacity + 1U) * sizeof(unsigned long long), stream);
  gpuMemsetAsync(keys, 0, (capacity + 1U) * sizeof(unsigned long long), stream);

  size_t n_blocks = (n_pixels + boost_block_size - 1) / boost_block_size;
  if (n_blocks > max_boost_blocks)
  {
    n_blocks = max_boost_blocks;
  }
  const dim3 block_size = {boost_block_size};
  const dim3 atomic_grid = {static_cast<unsigned int>(n_blocks)};
  island_boost_atomic_kernel<<<atomic_grid, block_size, 0, stream>>>(labels_per_pixel, n_pixels, baseline_flux,
                                                                    active_count, capacity, flux_per_label, keys);
  CHECK_LAST_GPU_ERROR();

  const dim3 finalize_grid = {static_cast<unsigned int>((capacity + boost_block_size - 1) / boost_block_size)};
  island_boost_finalize_kernel<<<finalize_grid, block_size, 0, stream>>>(keys, baseline_flux, active_count, capacity,
                                                                        flux_per_label, boost, found);
  CHECK_LAST_GPU_ERROR();

  gpuFreeAsync(keys, stream);
}

} // namespace breifast::gpu
