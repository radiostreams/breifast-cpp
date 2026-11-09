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

static constexpr size_t warp_size = 32;
static constexpr size_t boost_block_pixels = warp_size;
static constexpr size_t boost_block_labels = 8;
static constexpr size_t reduction_block_size = 256;
static constexpr size_t max_reduction_warps = (reduction_block_size + warp_size - 1) / warp_size;
static constexpr unsigned int full_warp_mask = 0xFFFFFFFFU;

struct IslandCandidate
{
  float abs_ratio = 0.F;
  float ratio = 0.F;
  size_t index = 0;
  bool found = false;
};

__device__ bool better(const IslandCandidate &candidate, const IslandCandidate &best)
{
  if (!candidate.found)
  {
    return false;
  }
  if (!best.found)
  {
    return true;
  }
  if (candidate.abs_ratio != best.abs_ratio)
  {
    return candidate.abs_ratio > best.abs_ratio;
  }
  return candidate.index < best.index;
}

__device__ IslandCandidate warp_reduce(IslandCandidate best, const unsigned int mask)
{
  for (int offset = static_cast<int>(warp_size) / 2; offset > 0; offset >>= 1)
  {
    IslandCandidate other;
    other.abs_ratio = __shfl_down_sync(mask, best.abs_ratio, offset);
    other.ratio = __shfl_down_sync(mask, best.ratio, offset);
    other.index = __shfl_down_sync(mask, best.index, offset);
    other.found = __shfl_down_sync(mask, best.found ? 1 : 0, offset) != 0;

    if (better(other, best))
    {
      best = other;
    }
  }
  return best;
}

__global__ void island_boost_candidates(const uint32_t *const labels_per_pixel, const size_t n_pixels,
                                        const size_t n_labels, const float *const baseline_flux,
                                        const uint32_t *const labels_to_process, const float *const flux_per_label,
                                        IslandCandidate *const candidates)
{
  const size_t pixel_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t label_id = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const bool valid = pixel_id < n_pixels && label_id < n_labels;

  IslandCandidate best;
  if (valid)
  {
    const auto label = labels_to_process != nullptr ? labels_to_process[label_id]
                                                   : static_cast<uint32_t>(label_id + 1U);
    if (label == labels_per_pixel[pixel_id])
    {
      const float ratio = flux_per_label[label_id] / baseline_flux[pixel_id];
      best.abs_ratio = fabsf(ratio);
      best.ratio = ratio;
      best.index = pixel_id;
      best.found = true;
    }
  }

  best = warp_reduce(best, full_warp_mask);

  if (threadIdx.x == 0 && label_id < n_labels)
  {
    candidates[label_id * gridDim.x + blockIdx.x] = best;
  }
}

__global__ void island_boost_reduction(const IslandCandidate *const candidates, const size_t n_labels,
                                       const size_t n_blocks, float *const boost, bool *const found)
{
  __shared__ IslandCandidate warp_candidates[max_reduction_warps];

  const size_t label = blockIdx.x;
  if (label >= n_labels)
  {
    return;
  }

  IslandCandidate best;
  for (size_t i = threadIdx.x; i < n_blocks; i += blockDim.x)
  {
    const IslandCandidate candidate = candidates[label * n_blocks + i];
    if (better(candidate, best))
    {
      best = candidate;
    }
  }

  best = warp_reduce(best, full_warp_mask);

  const size_t lane = threadIdx.x % warp_size;
  const size_t warp = threadIdx.x / warp_size;
  if (lane == 0)
  {
    warp_candidates[warp] = best;
  }
  __syncthreads();

  const size_t n_warps = (blockDim.x + warp_size - 1) / warp_size;
  if (warp == 0)
  {
    best = lane < n_warps ? warp_candidates[lane] : IslandCandidate{};
    best = warp_reduce(best, full_warp_mask);
    if (lane == 0)
    {
      boost[label] = best.found ? best.ratio : 0.F;
      found[label] = best.found;
    }
  }
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

void gpu_island_boost(const unsigned int *const labels_per_pixel, const size_t n_pixels, const size_t n_labels,
                      const float *const baseline_flux, const unsigned int *const labels_to_process,
                      const float *const flux_per_label, float *const boost, bool *const found, GpuStream stream)
{
  if (n_pixels == 0 || n_labels == 0)
  {
    throw std::runtime_error("gpu_island_boost: n_pixels and n_labels must be greater than 0");
  }

  const size_t n_blocks = (n_pixels + boost_block_pixels - 1) / boost_block_pixels;
  const size_t label_blocks = (n_labels + boost_block_labels - 1) / boost_block_labels;

  IslandCandidate *candidates = nullptr;
  gpuMallocAsync(reinterpret_cast<void **>(&candidates), n_labels * n_blocks * sizeof(IslandCandidate), stream);

  const dim3 candidate_block = {boost_block_pixels, boost_block_labels};
  const dim3 candidate_grid = {static_cast<unsigned int>(n_blocks), static_cast<unsigned int>(label_blocks)};
  island_boost_candidates<<<candidate_grid, candidate_block, 0, stream>>>(labels_per_pixel, n_pixels, n_labels,
                                                                         baseline_flux, labels_to_process,
                                                                         flux_per_label, candidates);
  CHECK_LAST_GPU_ERROR();

  const dim3 reduction_grid = {static_cast<unsigned int>(n_labels)};
  island_boost_reduction<<<reduction_grid, reduction_block_size, 0, stream>>>(candidates, n_labels, n_blocks, boost,
                                                                              found);
  CHECK_LAST_GPU_ERROR();

  gpuFreeAsync(candidates, stream);
}
} // namespace breifast::gpu
