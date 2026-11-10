#include <cfloat>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "gpu_pipeline.hpp"

#include "gpu_runtime.hpp"
#include "radix_keys.cuh"

namespace breifast::gpu
{
namespace
{
__global__ void donut_median_abs_kernel(const unsigned int *active_count, const float *stamps, size_t stamp_pixels,
                                        float scale, float *out_lmad)
{
  if (blockIdx.x >= *active_count)
  {
    return;
  }
  const auto i_stamp = blockIdx.x;
  const auto stamp = stamps + i_stamp * stamp_pixels;

  __shared__ uint32_t histogram[256];
  __shared__ uint32_t non_zero_count;
  __shared__ uint8_t first_bin;
  __shared__ uint32_t prefix;
  __shared__ uint32_t coverage;
  __shared__ uint32_t lower_max;
  for (size_t i = threadIdx.x; i < 256; i += blockDim.x)
  {
    histogram[i] = 0;
  }
  if (threadIdx.x == 0)
  {
    non_zero_count = 0;
    prefix = 0;
    coverage = 0;
    lower_max = 0;
  }
  __syncthreads();

  for (size_t i_pixel = threadIdx.x; i_pixel < stamp_pixels; i_pixel += blockDim.x)
  {
    const float value = stamp[i_pixel];
    if (value != 0.0f)
    {
      const uint8_t bin = static_cast<uint8_t>(abs_key(value) >> 24);
      atomicAdd(&histogram[bin], 1U);
      atomicAdd(&non_zero_count, 1U);
    }
  }
  __syncthreads();
  const auto mid = non_zero_count / 2;
  if (non_zero_count == 0)
  {
    if (threadIdx.x == 0)
    {
      out_lmad[i_stamp] = 0;
    }
    return;
  }

  size_t below = 0;
  if (threadIdx.x == 0)
  {
    for (size_t i = 0; i < 256; ++i)
    {
      const size_t count = histogram[i];
      if (below + count > mid)
      {
        first_bin = i;
        break;
      }
      below += count;
    }
    prefix = static_cast<uint32_t>(first_bin) << 24;
    coverage = 0xFFU << 24;
  }
  __syncthreads();

  for (size_t bin = 1; bin < 4; ++bin)
  {
    const auto shift = 16 - (bin - 1) * 8;
    for (size_t i = threadIdx.x; i < 256; i += blockDim.x)
    {
      histogram[i] = 0;
    }
    __syncthreads();

    for (size_t i_pixel = threadIdx.x; i_pixel < stamp_pixels; i_pixel += blockDim.x)
    {
      const float value = stamp[i_pixel];
      const uint32_t key = abs_key(value);
      if (value != 0.0f && (key & coverage) == prefix)
      {
        atomicAdd(&histogram[(key >> shift) & 0xFFU], 1U);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
      for (size_t i = 0; i < 256; ++i)
      {
        const size_t count = histogram[i];
        if (below + count > mid)
        {
          prefix |= static_cast<uint32_t>(i) << shift;
          coverage |= 0xFFU << shift;
          break;
        }
        below += count;
      }
    }
    __syncthreads();
  }

  if (non_zero_count % 2 == 0)
  {
    for (size_t i_pixel = threadIdx.x; i_pixel < stamp_pixels; i_pixel += blockDim.x)
    {
      const float value = stamp[i_pixel];
      const uint32_t key = abs_key(value);
      if (value != 0.0f && key < prefix)
      {
        atomicMax(&lower_max, key);
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    if (non_zero_count % 2 == 1)
    {
      out_lmad[i_stamp] = scale * abs_value(prefix);
    }
    else
    {
      const uint32_t lo_key = mid > below ? prefix : lower_max;
      out_lmad[i_stamp] = scale * (abs_value(lo_key) + abs_value(prefix)) / 2.F;
    }
  }
}

constexpr size_t stamp_block_threads = 256;

__global__ void stamp_abs_extreme_kernel(const unsigned int *active_count, const float *stamps, size_t stamp_pixels,
                                         float *out_extreme)
{
  if (blockIdx.x >= *active_count)
  {
    return;
  }
  const size_t i_stamp = blockIdx.x;
  const float *stamp = stamps + i_stamp * stamp_pixels;

  float min_value = FLT_MAX;
  float max_value = -FLT_MAX;
  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    const float value = stamp[i];
    min_value = fminf(min_value, value);
    max_value = fmaxf(max_value, value);
  }

  __shared__ float shared_min[stamp_block_threads];
  __shared__ float shared_max[stamp_block_threads];
  shared_min[threadIdx.x] = min_value;
  shared_max[threadIdx.x] = max_value;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (threadIdx.x < stride)
    {
      shared_min[threadIdx.x] = fminf(shared_min[threadIdx.x], shared_min[threadIdx.x + stride]);
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    out_extreme[i_stamp] = fmaxf(fabsf(shared_min[0]), fabsf(shared_max[0]));
  }
}

__global__ void stamp_extract_kernel(const unsigned int *active_count, const float *plane, size_t n_y, size_t n_x,
                                     const long *centre_y, const long *centre_x, long radius, long inner_boxsize,
                                     size_t stamp_pixels, float *stamps)
{
  if (blockIdx.x >= *active_count)
  {
    return;
  }
  const size_t stamp_id = blockIdx.x;
  const long side = 2 * radius + 1;
  const long half = inner_boxsize < radius ? inner_boxsize : radius;
  const long first_y = centre_y[stamp_id] - radius;
  const long first_x = centre_x[stamp_id] - radius;
  float *const out = stamps + stamp_id * stamp_pixels;

  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    const long row = static_cast<long>(i / static_cast<size_t>(side));
    const long col = static_cast<long>(i % static_cast<size_t>(side));
    const long plane_y = first_y + row;
    const long plane_x = first_x + col;
    const bool inside =
        plane_y >= 0 && plane_y < static_cast<long>(n_y) && plane_x >= 0 && plane_x < static_cast<long>(n_x);
    const long dr = row - radius;
    const long dc = col - radius;
    const bool inner = dr >= -half && dr <= half && dc >= -half && dc <= half;
    out[i] = inside && !inner ? plane[static_cast<size_t>(plane_y) * n_x + static_cast<size_t>(plane_x)] : 0.F;
  }
}

__global__ void stamp_extract_no_zero_kernel(const unsigned int *active_count, const float *plane, size_t n_y,
                                             size_t n_x, const long *centre_y, const long *centre_x, long radius,
                                             size_t stamp_pixels, float *stamps)
{
  if (blockIdx.x >= *active_count)
  {
    return;
  }
  const size_t stamp_id = blockIdx.x;
  const long side = 2 * radius + 1;
  const long first_y = centre_y[stamp_id] - radius;
  const long first_x = centre_x[stamp_id] - radius;
  float *const out = stamps + stamp_id * stamp_pixels;

  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    const long row = static_cast<long>(i / static_cast<size_t>(side));
    const long col = static_cast<long>(i % static_cast<size_t>(side));
    const long plane_y = first_y + row;
    const long plane_x = first_x + col;
    const bool inside =
        plane_y >= 0 && plane_y < static_cast<long>(n_y) && plane_x >= 0 && plane_x < static_cast<long>(n_x);
    out[i] = inside ? plane[static_cast<size_t>(plane_y) * n_x + static_cast<size_t>(plane_x)] : 0.F;
  }
}

constexpr size_t BlockDim = 256;

__global__ void donut_std_kernel(const unsigned int *active_count, const float *stamps, const size_t stamps_pixels,
                                 float *out_std)
{
  if (blockIdx.x >= *active_count)
  {
    return;
  }
  __shared__ double sum[BlockDim];
  __shared__ double sumsq[BlockDim];
  __shared__ uint32_t counts[BlockDim];
  sum[threadIdx.x] = 0.F;
  sumsq[threadIdx.x] = 0.F;
  counts[threadIdx.x] = 0;
  const float *stamp = stamps + blockIdx.x * stamps_pixels;

  __syncthreads();
  for (size_t i_pixel = threadIdx.x; i_pixel < stamps_pixels; i_pixel += blockDim.x)
  {
    const auto value = stamp[i_pixel];
    if (value != 0.F)
    {
      sum[threadIdx.x] += value;
      sumsq[threadIdx.x] += value * value;
      ++counts[threadIdx.x];
    }
  }
  __syncthreads();

  for (size_t factor = 1; factor < blockDim.x; factor *= 2)
  {
    if (threadIdx.x % (2 * factor) == 0)
    {
      sum[threadIdx.x] += sum[threadIdx.x + factor];
      sumsq[threadIdx.x] += sumsq[threadIdx.x + factor];
      counts[threadIdx.x] += counts[threadIdx.x + factor];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    if (counts[0] == 0)
    {
      out_std[blockIdx.x] = 0.F;
      return;
    }
    const auto mean = sum[0] / static_cast<float>(counts[0]);
    const auto meansq = sumsq[0] / static_cast<float>(counts[0]);
    const auto var = meansq - mean * mean;
    out_std[blockIdx.x] = sqrtf(static_cast<float>(var));
  }
}

__global__ void kernel_sum_kernel(const float *kernel, size_t n, float *out_sum)
{
  __shared__ float tmp[BlockDim];
  tmp[threadIdx.x] = 0.F;
  __syncthreads();

  for (size_t i = threadIdx.x; i < n; i += blockDim.x)
  {
    tmp[threadIdx.x] += kernel[i];
  }
  __syncthreads();
  for (size_t factor = 1; factor < blockDim.x; factor *= 2)
  {
    if (threadIdx.x % (2 * factor) == 0)
    {
      tmp[threadIdx.x] += tmp[threadIdx.x + factor];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    *out_sum = tmp[0];
  }
}

__global__ void stamp_beam_chi2_kernel(const unsigned int *active_count, const float *stamps, const float *kernel,
                                       const float *kernel_sum, const float *stamp_extreme,
                                       float inv_max_to_std_ratio, size_t stamp_pixels, float *out_chi2)
{
  if (blockIdx.x >= *active_count)
  {
    return;
  }
  const auto i_stamp = blockIdx.x;
  const float adjacent_noise = stamp_extreme[i_stamp] * inv_max_to_std_ratio;

  const float *stamp = stamps + i_stamp * stamp_pixels;

  __shared__ float tmp[BlockDim];
  tmp[threadIdx.x] = 0.F;
  __syncthreads();

  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    tmp[threadIdx.x] += stamp[i];
  }
  __syncthreads();
  for (size_t factor = 1; factor < blockDim.x; factor *= 2)
  {
    if (threadIdx.x % (2 * factor) == 0)
    {
      tmp[threadIdx.x] += tmp[threadIdx.x + factor];
    }
    __syncthreads();
  }
  const auto scale = tmp[0] / *kernel_sum;

  __shared__ uint32_t counts[BlockDim];
  counts[threadIdx.x] = 0U;
  tmp[threadIdx.x] = 0.F;
  __syncthreads();

  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    const auto value = stamp[i];
    if (value != 0.F)
    {
      const auto delta = value - scale * kernel[i];
      tmp[threadIdx.x] += delta * delta;
      ++counts[threadIdx.x];
    }
  }
  __syncthreads();

  for (size_t factor = 1; factor < blockDim.x; factor *= 2)
  {
    if (threadIdx.x % (2 * factor) == 0)
    {
      tmp[threadIdx.x] += tmp[threadIdx.x + factor];
      counts[threadIdx.x] += counts[threadIdx.x + factor];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    const auto count = counts[0];
    const auto resid2 = tmp[0];
    out_chi2[i_stamp] = (count > 0 && adjacent_noise != 0)
                            ? resid2 / (adjacent_noise * adjacent_noise * count)
                            : 0.F;
  }
}
} // namespace

/**
 * @brief Scaled median absolute value of the nonzero pixels of each stamp: the `lmad` of stamp_donut_stats.
 *
 * One block per stamp. The block keeps the pixels with `value != 0`, takes the median of `abs(value)` with a
 * block-level digit select (the even-count average of the two middle ranks), and writes `scale * median`, or 0 when
 * the stamp has no nonzero pixel. Device pointers; the work is enqueued on `stream`.
 * @param stamps `[n_stamps, stamp_pixels]` floats, row major, one stamp per row.
 * @param stamp_pixels `(2 * radius + 1)^2`.
 * @param scale Multiplier applied to the median, `1.4826` for the donut.
 * @param out_lmad `[n_stamps]` floats, written completely.
 */
void gpu_donut_median_abs(const float *stamps, const unsigned int *active_count, unsigned int capacity,
                          size_t stamp_pixels, float scale, float *out_lmad, GpuStream stream)
{
  if (capacity == 0)
  {
    throw std::runtime_error{"gpu_donut_median_abs: no stamps to process"};
  }
  if (active_count == nullptr)
  {
    throw std::runtime_error{"gpu_donut_median_abs: active_count must be non-null"};
  }
  if (stamp_pixels == 0)
  {
    throw std::runtime_error{"gpu_donut_median_abs: empty stamps"};
  }

  const size_t threads = 256;
  const dim3 block_size = {threads};
  const dim3 grid_size = {capacity};

  donut_median_abs_kernel<<<grid_size, block_size, 0, stream>>>(active_count, stamps, stamp_pixels, scale, out_lmad);
  CHECK_LAST_GPU_ERROR();
}

void gpu_donut_std(const float *stamps, const unsigned int *active_count, unsigned int capacity, size_t stamp_pixels,
                   float *out_std, GpuStream stream)
{
  if (capacity == 0)
  {
    throw std::runtime_error{"gpu_donut_std: no stamps to process"};
  }
  if (active_count == nullptr)
  {
    throw std::runtime_error{"gpu_donut_std: active_count must be non-null"};
  }
  if (stamp_pixels == 0)
  {
    throw std::runtime_error{"gpu_donut_std: empty stamps"};
  }

  const dim3 block_size = {BlockDim};
  const dim3 grid_size = {capacity};
  donut_std_kernel<<<grid_size, block_size, 0, stream>>>(active_count, stamps, stamp_pixels, out_std);
  CHECK_LAST_GPU_ERROR();
}

void gpu_stamp_abs_extreme(const float *stamps, const unsigned int *active_count, unsigned int capacity,
                           size_t stamp_pixels, float *out_extreme, GpuStream stream)
{
  if (capacity == 0)
  {
    throw std::runtime_error{"gpu_stamp_abs_extreme: no stamps to process"};
  }
  if (active_count == nullptr)
  {
    throw std::runtime_error{"gpu_stamp_abs_extreme: active_count must be non-null"};
  }
  if (stamp_pixels == 0)
  {
    throw std::runtime_error{"gpu_stamp_abs_extreme: empty stamps"};
  }

  const dim3 block_size = {stamp_block_threads};
  const dim3 grid_size = {capacity};
  stamp_abs_extreme_kernel<<<grid_size, block_size, 0, stream>>>(active_count, stamps, stamp_pixels, out_extreme);
  CHECK_LAST_GPU_ERROR();
}

void gpu_stamp_extract(const float *plane, size_t n_y, size_t n_x, const long *centre_y, const long *centre_x,
                       const unsigned int *active_count, unsigned int capacity, long radius, long inner_boxsize,
                       float *stamps, GpuStream stream)
{
  if (capacity == 0)
  {
    throw std::runtime_error{"gpu_stamp_extract: no stamps to process"};
  }
  if (active_count == nullptr)
  {
    throw std::runtime_error{"gpu_stamp_extract: active_count must be non-null"};
  }
  if (n_y == 0 || n_x == 0)
  {
    throw std::runtime_error{"gpu_stamp_extract: empty plane"};
  }
  if (radius < 0)
  {
    throw std::runtime_error{"gpu_stamp_extract: negative radius"};
  }
  if (inner_boxsize < 0)
  {
    throw std::runtime_error{"gpu_stamp_extract: negative inner_boxsize"};
  }

  const size_t side = static_cast<size_t>(2 * radius + 1);
  const size_t stamp_pixels = side * side;
  const dim3 block_size = {stamp_block_threads};
  const dim3 grid_size = {capacity};
  stamp_extract_kernel<<<grid_size, block_size, 0, stream>>>(active_count, plane, n_y, n_x, centre_y, centre_x, radius,
                                                             inner_boxsize, stamp_pixels, stamps);
  CHECK_LAST_GPU_ERROR();
}

void gpu_stamp_extract_no_zero(const float *plane, size_t n_y, size_t n_x, const long *centre_y, const long *centre_x,
                              const unsigned int *active_count, unsigned int capacity, long radius, float *stamps,
                              GpuStream stream)
{
  if (capacity == 0)
  {
    throw std::runtime_error{"gpu_stamp_extract_no_zero: no stamps to process"};
  }
  if (active_count == nullptr)
  {
    throw std::runtime_error{"gpu_stamp_extract_no_zero: active_count must be non-null"};
  }
  if (n_y == 0 || n_x == 0)
  {
    throw std::runtime_error{"gpu_stamp_extract_no_zero: empty plane"};
  }
  if (radius < 0)
  {
    throw std::runtime_error{"gpu_stamp_extract_no_zero: negative radius"};
  }

  const size_t side = static_cast<size_t>(2 * radius + 1);
  const size_t stamp_pixels = side * side;
  const dim3 block_size = {stamp_block_threads};
  const dim3 grid_size = {capacity};
  stamp_extract_no_zero_kernel<<<grid_size, block_size, 0, stream>>>(active_count, plane, n_y, n_x, centre_y, centre_x,
                                                                    radius, stamp_pixels, stamps);
  CHECK_LAST_GPU_ERROR();
}

void gpu_stamp_beam_chi2(const float *stamps, const float *kernel, const unsigned int *active_count,
                         unsigned int capacity, size_t stamp_pixels, const float *stamp_extreme,
                         float inv_max_to_std_ratio, float *out_chi2, GpuStream stream)
{
  if (capacity == 0)
  {
    throw std::runtime_error{"gpu_stamp_beam_chi2: no stamps to process"};
  }
  if (active_count == nullptr)
  {
    throw std::runtime_error{"gpu_stamp_beam_chi2: active_count must be non-null"};
  }
  if (stamp_pixels == 0)
  {
    throw std::runtime_error{"gpu_stamp_beam_chi2: empty stamps"};
  }

  thrust::device_vector<float> kernel_sum(1);
  kernel_sum_kernel<<<1, BlockDim, 0, stream>>>(kernel, stamp_pixels, thrust::raw_pointer_cast(kernel_sum.data()));
  CHECK_LAST_GPU_ERROR();

  const dim3 block_size = {BlockDim};
  const dim3 grid_size = {capacity};
  stamp_beam_chi2_kernel<<<grid_size, block_size, 0, stream>>>(
      active_count, stamps, kernel, thrust::raw_pointer_cast(kernel_sum.data()), stamp_extreme, inv_max_to_std_ratio,
      stamp_pixels, out_chi2);
  CHECK_LAST_GPU_ERROR();
}
} // namespace breifast::gpu