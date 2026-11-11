#include "gpu_kernels.hpp"

#include <cfloat>
#include <stdexcept>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "gpu_device_utils.cuh"

namespace breifast::gpu
{
namespace
{
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
