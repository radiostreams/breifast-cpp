#include "gpu_pipeline.hpp"

#include <cfloat>
#include <stdexcept>
#include <string>

#include "gpu_runtime.hpp"
#include "gpu_minmax.cuh"

namespace breifast::gpu
{
namespace
{
constexpr size_t label_stats_block_size = 256;

__global__ void label_stats_init_kernel(const unsigned int capacity, unsigned long long *const value_keys,
                                        float *const local_noise_max, unsigned int *const island_size)
{
  const auto label_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (label_id < capacity)
  {
    island_size[label_id] = 0;
    local_noise_max[label_id] = -FLT_MAX;
    value_keys[label_id] = 0;
  }
}

__global__ void label_stats_scan_kernel(const unsigned int *const labels, const float *const image,
                                        const float *const local_noise, const size_t n_pixels,
                                        const unsigned int capacity, unsigned long long *const value_keys,
                                        float *const local_noise_max, unsigned int *const island_size)
{
  for (size_t pixel_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; pixel_id < n_pixels;
       pixel_id += static_cast<size_t>(blockDim.x) * gridDim.x)
  {
    const auto label = labels[pixel_id];
    if (label == 0)
    {
      continue;
    }

    const auto slot = label - 1;
    atomicAdd(&island_size[slot], 1);

    const auto noise = local_noise[pixel_id];
    if (!isnan(noise))
    {
      atomic_max_float(&local_noise_max[slot], noise);
    }

    const auto abs_value = fabsf(image[pixel_id]);
    if (!isnan(abs_value))
    {
      const auto key = (uint64_t(__float_as_uint(abs_value)) << 32) | (0xFFFFFFFFu - uint32_t(pixel_id));
      atomicMax(&value_keys[slot], key);
    }
  }
}

__global__ void label_stats_finalize_kernel(const unsigned int *const active_count, const unsigned int capacity,
                                            const size_t n_x, const unsigned long long *const value_keys,
                                            const float *const local_noise_max, const unsigned int *const island_size,
                                            float *const value, float *const out_local_noise, long *const peak_y,
                                            long *const peak_x, unsigned int *const out_island_size)
{
  const unsigned int n_labels = *active_count;
  const unsigned int label = blockIdx.x * blockDim.x + threadIdx.x;
  if (label >= n_labels)
  {
    return;
  }

  const unsigned int size = island_size[label];
  out_island_size[label] = size;
  if (size == 0)
  {
    value[label] = -FLT_MAX;
    out_local_noise[label] = -FLT_MAX;
    peak_y[label] = 0;
    peak_x[label] = 0;
    return;
  }

  const unsigned long long key = value_keys[label];
  const unsigned int abs_bits = static_cast<unsigned int>(key >> 32);
  const unsigned int index = 0xFFFFFFFFU - static_cast<unsigned int>(key & 0xFFFFFFFFULL);
  value[label] = __uint_as_float(abs_bits);
  out_local_noise[label] = local_noise_max[label];
  peak_y[label] = static_cast<long>(index / n_x);
  peak_x[label] = static_cast<long>(index % n_x);
}
} // namespace

void gpu_label_stats(const unsigned int *const labels, const float *const image, const float *const local_noise,
                     const size_t n_pixels, const size_t n_x, const unsigned int *const active_count,
                     const unsigned int capacity, float *const value, float *const out_local_noise,
                     long *const peak_y, long *const peak_x, unsigned int *const island_size, GpuStream stream)
{
  if (n_pixels == 0)
  {
    throw std::runtime_error("gpu_label_stats: n_pixels must be greater than 0");
  }
  if (n_x == 0)
  {
    throw std::runtime_error("gpu_label_stats: n_x must be greater than 0");
  }
  if (active_count == nullptr)
  {
    throw std::runtime_error("gpu_label_stats: active_count must be non-null");
  }

  unsigned int host_count = 0;
  gpuMemcpyDeviceToHostAsync(&host_count, active_count, sizeof(unsigned int), stream);
  gpuStreamSynchronize(stream);
  if (host_count > capacity)
  {
    throw std::runtime_error("gpu_label_stats: active_count " + std::to_string(host_count) +
                             " exceeds capacity " + std::to_string(capacity));
  }
  if (host_count == 0)
  {
    return;
  }

  unsigned long long *value_keys = nullptr;
  float *local_noise_max = nullptr;
  unsigned int *sizes = nullptr;
  gpuMallocAsync(&value_keys, capacity * sizeof(unsigned long long), stream);
  gpuMallocAsync(&local_noise_max, capacity * sizeof(float), stream);
  gpuMallocAsync(&sizes, capacity * sizeof(unsigned int), stream);

  const dim3 block_size = {label_stats_block_size};
  const dim3 label_grid = {static_cast<unsigned int>((capacity + label_stats_block_size - 1) / label_stats_block_size)};
  label_stats_init_kernel<<<label_grid, block_size, 0, stream>>>(capacity, value_keys, local_noise_max, sizes);
  CHECK_LAST_GPU_ERROR();

  constexpr size_t max_blocks = 4096;
  size_t n_blocks = (n_pixels + label_stats_block_size - 1) / label_stats_block_size;
  if (n_blocks > max_blocks)
  {
    n_blocks = max_blocks;
  }
  const dim3 scan_grid = {static_cast<unsigned int>(n_blocks)};
  label_stats_scan_kernel<<<scan_grid, block_size, 0, stream>>>(labels, image, local_noise, n_pixels, capacity,
                                                                value_keys, local_noise_max, sizes);
  CHECK_LAST_GPU_ERROR();

  label_stats_finalize_kernel<<<label_grid, block_size, 0, stream>>>(
      active_count, capacity, n_x, value_keys, local_noise_max, sizes, value, out_local_noise, peak_y, peak_x,
      island_size);
  CHECK_LAST_GPU_ERROR();

  gpuFreeAsync(value_keys, stream);
  gpuFreeAsync(local_noise_max, stream);
  gpuFreeAsync(sizes, stream);
}
} // namespace breifast::gpu
