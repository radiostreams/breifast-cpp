#include <cfloat>
#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "gpu_kernels.hpp"

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

    for (size_t index = global_index(); index < n; index += static_cast<size_t>(blockDim.x) * gridDim.x)
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

__global__ void label_stats_init_kernel(const unsigned int capacity, unsigned long long *const value_keys,
                                        float *const local_noise_max, unsigned int *const island_size)
{
  const auto label_id = global_index();
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
  const size_t pixel_id = global_index();
  if (pixel_id >= n_pixels)
  {
    return;
  }
  const auto label = labels[pixel_id];
  if (label == 0)
  {
    return;
  }

  const auto slot = label - 1;
  atomicAdd(&island_size[slot], 1);

  const auto noise = local_noise[pixel_id];
  if (!isnan(noise))
  {
    atomic_update(&local_noise_max[slot], noise, FloatMax{});
  }

  const auto abs_value = fabsf(image[pixel_id]);
  if (!isnan(abs_value))
  {
    const auto key = (uint64_t(__float_as_uint(abs_value)) << 32) | (0xFFFFFFFFu - uint32_t(pixel_id));
    atomicMax(&value_keys[slot], key);
  }
}

__global__ void label_stats_finalize_kernel(const unsigned int *const active_count, const unsigned int capacity,
                                            const size_t n_x, const unsigned long long *const value_keys,
                                            const float *const local_noise_max, const unsigned int *const island_size,
                                            float *const value, float *const out_local_noise, long *const peak_y,
                                            long *const peak_x, unsigned int *const out_island_size)
{
  const unsigned int n_labels = *active_count;
  const auto label = global_index();
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

void gpu_label_islands(const bool *const mask, const size_t n_y, const size_t n_x, unsigned int *const labels,
                       unsigned int *const out_count, unsigned int *const scratch, GpuStream stream)
{
  if (n_y == 0 || n_x == 0)
  {
    throw std::runtime_error("gpu_label_islands: n_y and n_x must be greater than 0");
  }

  const size_t n = n_y * n_x;
  launch_per_element(init_labels_kernel, n, stream, mask, n, labels);

  thrust::device_vector<unsigned int> changed(1);
  launch_cooperative(converge_labels_kernel, n, stream, mask, n_y, n_x, labels,
                     thrust::raw_pointer_cast(changed.data()));

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

  thrust::device_vector<unsigned long long> value_keys(capacity);
  thrust::device_vector<float> local_noise_max(capacity);
  thrust::device_vector<unsigned int> sizes(capacity);
  unsigned long long *value_keys_ptr = thrust::raw_pointer_cast(value_keys.data());
  float *local_noise_max_ptr = thrust::raw_pointer_cast(local_noise_max.data());
  unsigned int *sizes_ptr = thrust::raw_pointer_cast(sizes.data());

  launch_per_element(label_stats_init_kernel, capacity, stream, capacity, value_keys_ptr, local_noise_max_ptr,
                     sizes_ptr);
  launch_per_element(label_stats_scan_kernel, n_pixels, stream, labels, image, local_noise, n_pixels, capacity,
                     value_keys_ptr, local_noise_max_ptr, sizes_ptr);
  launch_per_element(label_stats_finalize_kernel, capacity, stream, active_count, capacity, n_x, value_keys_ptr,
                     local_noise_max_ptr, sizes_ptr, value, out_local_noise, peak_y, peak_x, island_size);
}
} // namespace breifast::gpu
