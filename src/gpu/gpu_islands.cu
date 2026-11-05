#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "gpu_pipeline.hpp"

#include "gpu_runtime.hpp"

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

__global__ void relax_kernel(const bool *const mask, const size_t n_y, const size_t n_x, unsigned int *const labels,
                             unsigned int *const changed)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n = n_y * n_x;
  if (index >= n || !mask[index])
  {
    return;
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

  unsigned int *changed = nullptr;
  gpuMallocAsync(&changed, sizeof(unsigned int), stream);

  const size_t max_iterations = n;
  for (size_t iteration = 0; iteration < max_iterations; ++iteration)
  {
    gpuMemsetAsync(changed, 0, sizeof(unsigned int), stream);
    relax_kernel<<<grid_size, block_size, 0, stream>>>(mask, n_y, n_x, labels, changed);
    CHECK_LAST_GPU_ERROR();

    unsigned int host_changed = 0;
    gpuMemcpyDeviceToHostAsync(&host_changed, changed, sizeof(unsigned int), stream);
    gpuStreamSynchronize(stream);
    if (host_changed == 0)
    {
      break;
    }
  }

  gpuFreeAsync(changed, stream);

  mark_roots_kernel<<<grid_size, block_size, 0, stream>>>(n, labels, scratch);
  CHECK_LAST_GPU_ERROR();
  thrust::inclusive_scan(thrust::cuda::par.on(stream), scratch, scratch + n, scratch);

  scatter_labels_kernel<<<grid_size, block_size, 0, stream>>>(mask, n, labels, scratch, out_count);
  CHECK_LAST_GPU_ERROR();
}
} // namespace breifast::gpu
