#include <cassert>
#include <cfloat>
#include <climits>

#include <cooperative_groups.h>
#include <cub/device/device_scan.cuh>

#include "gpu_band.hpp"
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

__global__ void island_boost_finalize_kernel(const unsigned long long *const keys, const float *const baseline_flux,
                                             const unsigned int *const active_count, const unsigned int capacity,
                                             const float *const flux_per_label, float *const boost)
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
    return;
  }

  const unsigned long long key = keys[label];
  if (key == 0ULL) // no pixel carries the label
  {
    boost[slot] = 0.F;
    return;
  }

  const unsigned int index = 0xFFFFFFFFU - static_cast<unsigned int>(key & 0xFFFFFFFFULL);
  boost[slot] = flux_per_label[slot] / baseline_flux[index];
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

void LabelScratch::ensure(const std::size_t n_pixels)
{
  assert(n_pixels <= static_cast<std::size_t>(INT_MAX));
  root_ranks.ensure(n_pixels);
  std::size_t bytes = 0;
  CHECK_GPU(cub::DeviceScan::InclusiveSum(nullptr, bytes, root_ranks.data(), root_ranks.data(),
                                          static_cast<int>(n_pixels)));
  scan_storage.ensure(bytes);
}

void gpu_label_islands(const bool *const mask, const size_t n_y, const size_t n_x, unsigned int *const labels,
                       unsigned int *const out_count, LabelScratch &scratch, cudaStream_t stream)
{
  assert(n_y > 0 && n_x > 0);

  const size_t n = n_y * n_x;
  assert(scratch.root_ranks.size() >= n);
  launch_per_element(init_labels_kernel, n, stream, mask, n, labels);
  launch_cooperative(converge_labels_kernel, n, stream, mask, n_y, n_x, labels, scratch.changed.data());

  // Rank each component's root among the roots: an inclusive scan, in place.
  unsigned int *const ranks = scratch.root_ranks.data();
  launch_per_element(mark_roots_kernel, n, stream, n, labels, ranks);
  std::size_t scan_bytes = scratch.scan_storage.size();
  CHECK_GPU(cub::DeviceScan::InclusiveSum(scratch.scan_storage.data(), scan_bytes, ranks, ranks, static_cast<int>(n),
                                          stream));
  launch_per_element(scatter_labels_kernel, n, stream, mask, n, labels, ranks, out_count);
}

void gpu_island_boost(const unsigned int *const labels_per_pixel, const float *const baseline_flux,
                      const size_t n_pixels, const unsigned int *const active_count, const unsigned int capacity,
                      const float *const flux_per_label, float *const boost, IslandScratch &scratch,
                      cudaStream_t stream)
{
  assert(n_pixels > 0);
  assert(active_count != nullptr);
  assert(capacity > 0);

  assert(scratch.boost_keys.size() >= capacity + 1U);
  unsigned long long *const keys_ptr = scratch.boost_keys.data();
  // atomicMax keys: they start at 0, and the scratch still holds the previous call's.
  CHECK_GPU(cudaMemsetAsync(keys_ptr, 0, (capacity + 1U) * sizeof(unsigned long long), stream));

  launch_per_element(island_boost_atomic_kernel, n_pixels, stream, labels_per_pixel, n_pixels, baseline_flux,
                     active_count, capacity, flux_per_label, keys_ptr);
  launch_per_element(island_boost_finalize_kernel, capacity, stream, keys_ptr, baseline_flux, active_count, capacity,
                     flux_per_label, boost);
}

void gpu_label_stats(const unsigned int *const labels, const float *const image, const float *const local_noise,
                     const size_t n_pixels, const size_t n_x, const unsigned int *const active_count,
                     const unsigned int capacity, float *const value, float *const out_local_noise, long *const peak_y,
                     long *const peak_x, unsigned int *const island_size, IslandScratch &scratch, cudaStream_t stream)
{
  assert(n_pixels > 0);
  assert(n_x > 0);
  assert(active_count != nullptr);

  unsigned int host_count = 0;
  CHECK_GPU(cudaMemcpyAsync(&host_count, active_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaStreamSynchronize(stream));
  assert(host_count <= capacity);
  if (host_count == 0)
  {
    return;
  }

  // The init kernel below resets all three for this call.
  assert(scratch.sizes.size() >= capacity);
  unsigned long long *const value_keys_ptr = scratch.value_keys.data();
  float *const local_noise_max_ptr = scratch.local_noise_max.data();
  unsigned int *const sizes_ptr = scratch.sizes.data();

  launch_per_element(label_stats_init_kernel, capacity, stream, capacity, value_keys_ptr, local_noise_max_ptr,
                     sizes_ptr);
  launch_per_element(label_stats_scan_kernel, n_pixels, stream, labels, image, local_noise, n_pixels, capacity,
                     value_keys_ptr, local_noise_max_ptr, sizes_ptr);
  launch_per_element(label_stats_finalize_kernel, capacity, stream, active_count, capacity, n_x, value_keys_ptr,
                     local_noise_max_ptr, sizes_ptr, value, out_local_noise, peak_y, peak_x, island_size);
}
} // namespace breifast::gpu
