#include <cassert>
#include <cfloat>

#include <cuda/std/functional>

#include "gpu_kernels.hpp"

#include "gpu_device_utils.cuh"
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

  __shared__ uint32_t histogram[radix_bins];
  __shared__ uint32_t non_zero_count;
  __shared__ uint32_t below;
  __shared__ uint32_t prefix;
  __shared__ uint32_t coverage;
  __shared__ uint32_t lower_max;
  if (threadIdx.x == 0)
  {
    below = 0;
    prefix = 0;
    coverage = 0;
    lower_max = 0;
  }

  // One byte of the nonzero pixels' abs keys per pass. coverage starts at 0, so the first pass sees every nonzero pixel
  // and its histogram sums to their count.
  for (int shift = 24; shift >= 0; shift -= 8)
  {
    for (size_t i = threadIdx.x; i < radix_bins; i += blockDim.x)
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
      if (shift == 24)
      {
        non_zero_count = 0;
        for (uint32_t i = 0; i < radix_bins; ++i)
        {
          non_zero_count += histogram[i];
        }
      }
      if (non_zero_count > 0)
      {
        select_radix_digit(histogram, non_zero_count / 2, shift, below, prefix, coverage);
      }
    }
    __syncthreads();

    if (non_zero_count == 0)
    {
      if (threadIdx.x == 0)
      {
        out_lmad[i_stamp] = 0;
      }
      return;
    }
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
    out_lmad[i_stamp] = scale * median_of_keys(non_zero_count, below, prefix, lower_max, true);
  }
}

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

  min_value = block_reduce(min_value, FloatMin{});
  max_value = block_reduce(max_value, FloatMax{});

  if (threadIdx.x == 0)
  {
    out_extreme[i_stamp] = fmaxf(fabsf(min_value), fabsf(max_value));
  }
}

template <bool skip_inner> __global__ void stamp_extract_kernel(const unsigned int *active_count, const float *plane,
                                                                   size_t n_y, size_t n_x, const long *centre_y,
                                                                   const long *centre_x, long radius, long inner_boxsize,
                                                                   size_t stamp_pixels, float *stamps)
{
  if (blockIdx.x >= *active_count)
  {
    return;
  }
  const size_t stamp_id = blockIdx.x;
  const long side = 2 * radius + 1;
  const long half = skip_inner ? (inner_boxsize < radius ? inner_boxsize : radius) : -1;
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
    const bool inner = skip_inner && dr >= -half && dr <= half && dc >= -half && dc <= half;
    out[i] = inside && !inner ? plane[static_cast<size_t>(plane_y) * n_x + static_cast<size_t>(plane_x)] : 0.F;
  }
}

/// Sum, sum of squares and count of the nonzero pixels of a stamp.
struct Moments
{
  double sum;
  double sumsq;
  uint32_t count;
};

__host__ __device__ Moments operator+(const Moments &a, const Moments &b)
{
  return {a.sum + b.sum, a.sumsq + b.sumsq, a.count + b.count};
}

__global__ void donut_std_kernel(const unsigned int *active_count, const float *stamps, const size_t stamps_pixels,
                                 float *out_std)
{
  if (blockIdx.x >= *active_count)
  {
    return;
  }
  const float *stamp = stamps + blockIdx.x * stamps_pixels;

  Moments moments{0.0, 0.0, 0U};
  for (size_t i_pixel = threadIdx.x; i_pixel < stamps_pixels; i_pixel += blockDim.x)
  {
    const auto value = stamp[i_pixel];
    if (value != 0.F)
    {
      moments.sum += value;
      moments.sumsq += value * value;
      ++moments.count;
    }
  }

  const Moments total = block_reduce(moments, cuda::std::plus<>{});

  if (threadIdx.x == 0)
  {
    if (total.count == 0)
    {
      out_std[blockIdx.x] = 0.F;
      return;
    }
    const auto mean = total.sum / static_cast<float>(total.count);
    const auto meansq = total.sumsq / static_cast<float>(total.count);
    const auto var = meansq - mean * mean;
    out_std[blockIdx.x] = sqrtf(static_cast<float>(var));
  }
}

__global__ void stamp_beam_chi2_kernel(const unsigned int *active_count, const float *stamps, const float *kernel,
                                       const float kernel_sum, const float *stamp_extreme,
                                       float inv_max_to_std_ratio, size_t stamp_pixels, float *out_chi2)
{
  if (blockIdx.x >= *active_count)
  {
    return;
  }
  const auto i_stamp = blockIdx.x;
  const float adjacent_noise = stamp_extreme[i_stamp] * inv_max_to_std_ratio;

  const float *stamp = stamps + i_stamp * stamp_pixels;

  float stamp_sum = 0.F;
  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    stamp_sum += stamp[i];
  }
  stamp_sum = block_reduce(stamp_sum, cuda::std::plus<>{});
  const auto scale = stamp_sum / kernel_sum;

  float resid2 = 0.F;
  uint32_t count = 0U;
  for (size_t i = threadIdx.x; i < stamp_pixels; i += blockDim.x)
  {
    const auto value = stamp[i];
    if (value != 0.F)
    {
      const auto delta = value - scale * kernel[i];
      resid2 += delta * delta;
      ++count;
    }
  }
  resid2 = block_reduce(resid2, cuda::std::plus<>{});
  count = block_reduce(count, cuda::std::plus<>{});
  if (threadIdx.x == 0)
  {
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
                          size_t stamp_pixels, float scale, float *out_lmad, cudaStream_t stream)
{
  assert(capacity > 0);
  assert(active_count != nullptr);
  assert(stamp_pixels > 0);

  launch_per_island(donut_median_abs_kernel, capacity, stream, active_count, stamps, stamp_pixels, scale, out_lmad);
}

void gpu_donut_std(const float *stamps, const unsigned int *active_count, unsigned int capacity, size_t stamp_pixels,
                   float *out_std, cudaStream_t stream)
{
  assert(capacity > 0);
  assert(active_count != nullptr);
  assert(stamp_pixels > 0);

  launch_per_island(donut_std_kernel, capacity, stream, active_count, stamps, stamp_pixels, out_std);
}

void gpu_stamp_abs_extreme(const float *stamps, const unsigned int *active_count, unsigned int capacity,
                           size_t stamp_pixels, float *out_extreme, cudaStream_t stream)
{
  assert(capacity > 0);
  assert(active_count != nullptr);
  assert(stamp_pixels > 0);

  launch_per_island(stamp_abs_extreme_kernel, capacity, stream, active_count, stamps, stamp_pixels, out_extreme);
}

void gpu_stamp_extract(const float *plane, size_t n_y, size_t n_x, const long *centre_y, const long *centre_x,
                       const unsigned int *active_count, unsigned int capacity, long radius, long inner_boxsize,
                       float *stamps, cudaStream_t stream)
{
  assert(capacity > 0);
  assert(active_count != nullptr);
  assert(n_y > 0 && n_x > 0);
  assert(radius >= 0);
  assert(inner_boxsize >= 0);

  const size_t side = static_cast<size_t>(2 * radius + 1);
  const size_t stamp_pixels = side * side;
  launch_per_island(stamp_extract_kernel<true>, capacity, stream, active_count, plane, n_y, n_x, centre_y, centre_x,
                    radius, inner_boxsize, stamp_pixels, stamps);
}

void gpu_stamp_extract_no_zero(const float *plane, size_t n_y, size_t n_x, const long *centre_y, const long *centre_x,
                               const unsigned int *active_count, unsigned int capacity, long radius, float *stamps,
                               cudaStream_t stream)
{
  assert(capacity > 0);
  assert(active_count != nullptr);
  assert(n_y > 0 && n_x > 0);
  assert(radius >= 0);

  const size_t side = static_cast<size_t>(2 * radius + 1);
  const size_t stamp_pixels = side * side;
  launch_per_island(stamp_extract_kernel<false>, capacity, stream, active_count, plane, n_y, n_x, centre_y, centre_x,
                    radius, 0, stamp_pixels, stamps);
}

void gpu_stamp_beam_chi2(const float *stamps, const float *kernel, const float kernel_sum,
                         const unsigned int *active_count, unsigned int capacity, size_t stamp_pixels,
                         const float *stamp_extreme, float inv_max_to_std_ratio, float *out_chi2, cudaStream_t stream)
{
  assert(capacity > 0);
  assert(active_count != nullptr);
  assert(stamp_pixels > 0);

  launch_per_island(stamp_beam_chi2_kernel, capacity, stream, active_count, stamps, kernel, kernel_sum,
                    stamp_extreme, inv_max_to_std_ratio, stamp_pixels, out_chi2);
}
} // namespace breifast::gpu