#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>

#include <cub/device/device_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "gpu_band.hpp"
#include "gpu_kernels.hpp"

#include "gpu_device_utils.cuh"

namespace breifast::gpu
{

namespace
{
__global__ void init_labels_kernel(const bool *const mask, const size_t n, unsigned int *const labels)
{
  const size_t index = global_index();
  if (index < n)
  {
    labels[index] = mask[index] ? static_cast<unsigned int>(index) : background_label;
  }
}

// The labels form a forest during labelling: a masked pixel's label is the flat index of its parent, a smaller index
// or its own, and a root is a pixel that is its own parent. Every write keeps a parent below its child, so following
// the parents always ends.

/// Root of the tree of masked pixel `pixel`: follows the parents until a pixel whose label is its own index. Reads
/// from L2: other blocks rewrite parents meanwhile, and a stale L1 copy would only lengthen the walk.
__device__ unsigned int find_root(const unsigned int *const labels, unsigned int pixel)
{
  unsigned int parent = __ldcg(&labels[pixel]);
  while (parent != pixel)
  {
    pixel = parent;
    parent = __ldcg(&labels[pixel]);
  }
  return pixel;
}

/// Puts masked pixels `i` and `j` in the same tree: hooks the larger of their roots under the smaller, so that each
/// component's root ends as its smallest index whatever the order of the joins. If another thread re-hooked that root
/// between the walks and the atomic, `old` is where it had put it, and the loop joins that tree too.
__device__ void join(unsigned int *const labels, unsigned int i, unsigned int j)
{
  while (true)
  {
    auto root_i = find_root(labels, i);
    auto root_j = find_root(labels, j);
    if (root_i == root_j)
    {
      return;
    }

    if (root_i > root_j)
    {
      root_i = root_i ^ root_j;
      root_j = root_i ^ root_j;
      root_i = root_i ^ root_j;
    }

    const auto old = atomicMin(&labels[root_j], root_i);
    if (old == root_j)
    {
      return;
    }

    i = root_i;
    j = old;
  }
}

/// Pixels per step of the union-find kernels: one 16-byte load of the mask, skipped whole when all background.
constexpr unsigned int mask_group = 16;

/// Upper bound on the grid of the union-find kernels: the threads stride over the mask, so that the blocks' cost does
/// not grow with the plane, whose pixels are mostly background.
constexpr unsigned int union_find_max_blocks = 1024;

/// Calls `visit(i)` for every masked pixel i of [0, n), striding over the mask 16 bytes at a time. `mask` must be
/// 16-byte aligned.
template <typename Visit>
__device__ void for_each_masked(const bool *const mask, const unsigned int n, const Visit &visit)
{
  const unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = gridDim.x * blockDim.x;
  const unsigned int groups = n / mask_group;
  for (unsigned int group = first; group < groups; group += stride)
  {
    const uint4 bytes = reinterpret_cast<const uint4 *>(mask)[group];
    if ((bytes.x | bytes.y | bytes.z | bytes.w) == 0U)
    {
      continue;
    }
    const unsigned int words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
#pragma unroll
    for (unsigned int k = 0; k < mask_group; ++k)
    {
      if (((words[k / 4] >> (8 * (k % 4))) & 0xFFU) != 0U)
      {
        visit(group * mask_group + k);
      }
    }
  }
  for (unsigned int i = groups * mask_group + first; i < n; i += stride) // the last n % 16 pixels
  {
    if (mask[i])
    {
      visit(i);
    }
  }
}

/// Union-find step of gpu_label_islands: each masked pixel joins its left and upper masked neighbours (the right and
/// lower links are theirs). Only the masked pixels read labels.
__global__ void merge_labels_kernel(const bool *const mask, const unsigned int n_y, const unsigned int n_x,
                                    unsigned int *const labels)
{
  for_each_masked(mask, n_y * n_x,
                  [&](const unsigned int i)
                  {
                    if (i % n_x > 0 && mask[i - 1])
                    {
                      join(labels, i, i - 1);
                    }
                    if (i >= n_x && mask[i - n_x])
                    {
                      join(labels, i, i - n_x);
                    }
                  });
}

/// Last step of the union-find: every masked pixel's label becomes its root, its component's smallest index.
__global__ void flatten_labels_kernel(const bool *const mask, const unsigned int n, unsigned int *const labels)
{
  for_each_masked(mask, n, [&](const unsigned int i) { labels[i] = find_root(labels, i); });
}

__global__ void scatter_labels_kernel(const bool *const mask, const size_t n, unsigned int *const labels,
                                      const unsigned int *const root_ranks)
{
  const size_t i = global_index();
  if (i < n)
  {
    labels[i] = mask[i] ? root_ranks[labels[i]] : 0U;
  }
}

__global__ void island_boost_atomic_kernel(const unsigned int *const labels_per_pixel, const size_t n_pixels,
                                           const float *const baseline_flux, const unsigned int n_labels,
                                           const float *const flux_per_label, unsigned long long *const keys)
{
  const size_t index = global_index();
  if (index >= n_pixels)
  {
    return;
  }
  const unsigned int label = labels_per_pixel[index];
  if (label == 0U || label > n_labels)
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
                                             const unsigned int n_labels, const float *const flux_per_label,
                                             float *const boost)
{
  const auto slot = global_index();
  if (slot >= n_labels)
  {
    return;
  }

  const unsigned int label = static_cast<unsigned int>(slot) + 1U;
  const unsigned long long key = keys[label];
  if (key == 0ULL) // no pixel carries the label
  {
    boost[slot] = 0.F;
    return;
  }

  const unsigned int index = 0xFFFFFFFFU - static_cast<unsigned int>(key & 0xFFFFFFFFULL);
  boost[slot] = flux_per_label[slot] / baseline_flux[index];
}

__global__ void label_stats_init_kernel(const unsigned int n_labels, unsigned long long *const value_keys,
                                        float *const local_noise_max, unsigned int *const island_size)
{
  const auto label_id = global_index();
  if (label_id < n_labels)
  {
    island_size[label_id] = 0;
    local_noise_max[label_id] = -FLT_MAX;
    value_keys[label_id] = 0;
  }
}

__global__ void label_stats_scan_kernel(const unsigned int *const labels, const float *const image,
                                        const float *const local_noise, const size_t n_pixels,
                                        unsigned long long *const value_keys, float *const local_noise_max,
                                        unsigned int *const island_size)
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

__global__ void label_stats_finalize_kernel(const unsigned int n_labels, const size_t n_x,
                                            const unsigned long long *const value_keys,
                                            const float *const local_noise_max, const unsigned int *const island_size,
                                            float *const value, float *const out_local_noise, long *const peak_y,
                                            long *const peak_x, unsigned int *const out_island_size)
{
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

/// Input of the root scan: 1 where pixel i is the root of its component (its label is its own index), else 0.
struct IsRoot
{
  const unsigned int *labels;
  __host__ __device__ unsigned int operator()(const unsigned int i) const
  {
    return labels[i] == i ? 1U : 0U;
  }
};

/// Output of the root scan: keeps the inclusive count only at roots, as the component's rank (1-based), and the total
/// at the last pixel. Its own reference type: `out[i] = rank` writes through operator=.
struct RootRankOutput
{
  using value_type = unsigned int;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = RootRankOutput;
  using iterator_category = std::random_access_iterator_tag;

  const unsigned int *labels;
  unsigned int *ranks;
  unsigned int *out_count;
  unsigned int last; // n - 1
  unsigned int i = 0;

  __host__ __device__ RootRankOutput operator+(const difference_type k) const
  {
    RootRankOutput moved = *this;
    moved.i += static_cast<unsigned int>(k);
    return moved;
  }
  __host__ __device__ RootRankOutput operator[](const difference_type k) const
  {
    return *this + k;
  }
  __host__ __device__ RootRankOutput operator*() const
  {
    return *this;
  }

  __device__ const RootRankOutput &operator=(const unsigned int rank) const
  {
    if (labels[i] == i)
    {
      ranks[i] = rank;
    }
    if (i == last)
    {
      *out_count = rank;
    }
    return *this;
  }
};

} // namespace

void LabelScratch::ensure(const std::size_t n_pixels)
{
  assert(n_pixels <= static_cast<std::size_t>(INT_MAX));
  root_ranks.ensure(n_pixels);
  std::size_t bytes = 0;
  // Sized with gpu_label_islands' iterator types; nothing is dereferenced.
  const auto flags = thrust::make_transform_iterator(thrust::make_counting_iterator(0U), IsRoot{nullptr});
  CHECK_GPU(cub::DeviceScan::InclusiveSum(nullptr, bytes, flags, RootRankOutput{nullptr, nullptr, nullptr, 0U},
                                          static_cast<int>(n_pixels)));
  scan_storage.ensure(bytes);
}

void gpu_label_islands(const bool *const mask, const size_t n_y, const size_t n_x, unsigned int *const labels,
                       unsigned int *const out_count, LabelScratch &scratch, cudaStream_t stream)
{
  assert(n_y > 0 && n_x > 0);

  const size_t n = n_y * n_x;
  assert(scratch.root_ranks.size() >= n);
  // Union-find: join the neighbouring masked pixels' trees, then point every masked pixel at its root.
  assert(reinterpret_cast<std::uintptr_t>(mask) % alignof(uint4) == 0); // whole buffers: 256-byte aligned
  const std::size_t groups = (n + mask_group - 1) / mask_group;
  const auto union_find_grid =
      static_cast<unsigned int>(std::min<std::size_t>((groups + block_size - 1) / block_size, union_find_max_blocks));
  launch(merge_labels_kernel, union_find_grid, block_size, stream, mask, static_cast<unsigned int>(n_y),
         static_cast<unsigned int>(n_x), labels);
  launch(flatten_labels_kernel, union_find_grid, block_size, stream, mask, static_cast<unsigned int>(n), labels);

  // Rank each component's root among the roots: an inclusive scan of the root flags that keeps the count at the roots
  // only, and the total in out_count.
  unsigned int *const ranks = scratch.root_ranks.data();
  const auto root_flags = thrust::make_transform_iterator(thrust::make_counting_iterator(0U), IsRoot{labels});
  const auto root_ranks_output = RootRankOutput{labels, ranks, out_count, static_cast<unsigned int>(n - 1)};
  std::size_t scan_bytes = scratch.scan_storage.size();
  CHECK_GPU(cub::DeviceScan::InclusiveSum(scratch.scan_storage.data(), scan_bytes, root_flags, root_ranks_output,
                                          static_cast<int>(n), stream));
}

void gpu_number_islands(const bool *const mask, const size_t n, unsigned int *const labels, const LabelScratch &scratch,
                        cudaStream_t stream)
{
  assert(n > 0);
  assert(scratch.root_ranks.size() >= n);
  launch_per_element(scatter_labels_kernel, n, stream, mask, n, labels, scratch.root_ranks.data());
}

void gpu_init_labels(const bool *const mask, const size_t n, unsigned int *const labels, cudaStream_t stream)
{
  launch_per_element(init_labels_kernel, n, stream, mask, n, labels);
}

void gpu_island_boost(const unsigned int *const labels_per_pixel, const float *const baseline_flux,
                      const size_t n_pixels, const unsigned int n_labels, const float *const flux_per_label,
                      float *const boost, IslandScratch &scratch, cudaStream_t stream)
{
  assert(n_pixels > 0);
  assert(n_labels > 0);

  assert(scratch.boost_keys.size() >= n_labels + 1U);
  unsigned long long *const keys_ptr = scratch.boost_keys.data();
  // atomicMax keys: they start at 0, and the scratch still holds the previous call's.
  CHECK_GPU(cudaMemsetAsync(keys_ptr, 0, (n_labels + 1U) * sizeof(unsigned long long), stream));

  launch_per_element(island_boost_atomic_kernel, n_pixels, stream, labels_per_pixel, n_pixels, baseline_flux, n_labels,
                     flux_per_label, keys_ptr);
  launch_per_element(island_boost_finalize_kernel, n_labels, stream, keys_ptr, baseline_flux, n_labels, flux_per_label,
                     boost);
}

void gpu_label_stats(const unsigned int *const labels, const float *const image, const float *const local_noise,
                     const size_t n_pixels, const size_t n_x, const unsigned int n_labels, float *const value,
                     float *const out_local_noise, long *const peak_y, long *const peak_x,
                     unsigned int *const island_size, IslandScratch &scratch, cudaStream_t stream)
{
  assert(n_pixels > 0);
  assert(n_x > 0);
  assert(n_labels > 0);

  // The init kernel below resets all three for this call.
  assert(scratch.sizes.size() >= n_labels);
  unsigned long long *const value_keys_ptr = scratch.value_keys.data();
  float *const local_noise_max_ptr = scratch.local_noise_max.data();
  unsigned int *const sizes_ptr = scratch.sizes.data();

  launch_per_element(label_stats_init_kernel, n_labels, stream, n_labels, value_keys_ptr, local_noise_max_ptr,
                     sizes_ptr);
  launch_per_element(label_stats_scan_kernel, n_pixels, stream, labels, image, local_noise, n_pixels, value_keys_ptr,
                     local_noise_max_ptr, sizes_ptr);
  launch_per_element(label_stats_finalize_kernel, n_labels, stream, n_labels, n_x, value_keys_ptr, local_noise_max_ptr,
                     sizes_ptr, value, out_local_noise, peak_y, peak_x, island_size);
}
} // namespace breifast::gpu
