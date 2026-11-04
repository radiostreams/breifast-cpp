#include "gpu_pipeline.hpp"

#include "gpu_runtime.hpp"
#include "radix_keys.cuh"

namespace breifast::gpu
{
namespace
{
__global__ void donut_median_abs_kernel(const float *stamps, size_t stamp_pixels, float scale, float *out_lmad)
{
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
void gpu_donut_median_abs(const float *stamps, size_t n_stamps, size_t stamp_pixels, float scale, float *out_lmad,
                          GpuStream stream)
{
  if (n_stamps == 0)
  {
    throw std::runtime_error{"gpu_donut_median_abs: no stamps to process"};
  }
  if (stamp_pixels == 0)
  {
    throw std::runtime_error{"gpu_donut_median_abs: empty stamps"};
  }

  const size_t threads = 256;
  const dim3 block_size = {threads};
  const dim3 grid_size = {static_cast<unsigned int>(n_stamps)};

  donut_median_abs_kernel<<<grid_size, block_size, 0, stream>>>(stamps, stamp_pixels, scale, out_lmad);
  CHECK_LAST_GPU_ERROR();
}
} // namespace breifast::gpu