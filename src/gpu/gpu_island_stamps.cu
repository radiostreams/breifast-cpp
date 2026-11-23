#include <cassert>
#include <cfloat>

#include <cuda/std/functional>

#include "gpu_kernels.hpp"

#include "gpu_device_utils.cuh"
#include "radix_keys.cuh"
#include "radix_select.cuh"

namespace breifast::gpu
{
namespace
{

// One radix select pass over the nonzero pixels of every stamp, for gpu_donut_median_abs: stamp_blocks
// consecutive blocks per stamp, striding over its pixels. A block counts the digit at `shift` of the abs keys that
// match the prefix into a histogram of its own in shared memory, then adds it to its stamp's. The last block of a stamp
// picks the digit for the next pass (finish_radix_pass), and after the last pass writes `scale * median`. The first
// pass sees every nonzero pixel, so its last block also records their count, the median's n; a stamp with none gets 0
// and the later passes skip it.
__global__ void donut_median_abs_kernel(const StampWindows windows, int shift, uint32_t mask, float scale,
                                        DonutMedianState *states, float *out_lmad)
{
  const unsigned int i_stamp = blockIdx.x / stamp_blocks;
  const unsigned int i_block = blockIdx.x % stamp_blocks;
  DonutMedianState &state = states[i_stamp];
  const bool first_pass = shift == select_shifts[0];
  if (!first_pass && state.count == 0)
  {
    return;
  }

  __shared__ uint32_t local[select_bins];
  for (uint32_t bin = threadIdx.x; bin < select_bins; bin += blockDim.x)
  {
    local[bin] = 0;
  }
  __syncthreads();

  const uint32_t below = state.select[0];
  const uint32_t prefix = state.select[1];
  const uint32_t coverage = state.select[2];
  // Last pass, even count: also the largest key below the 22-bit prefix, for rank count/2 - 1.
  const bool track_lower = shift == 0 && state.count % 2 == 0;
  uint32_t local_lower = 0;

  const Stamp stamp(windows, i_stamp);
  const size_t stamp_pixels = stamp.size();
  const size_t stride = size_t{stamp_blocks} * blockDim.x;
  for (size_t i = size_t{i_block} * blockDim.x + threadIdx.x; i < stamp_pixels; i += stride)
  {
    const float value = stamp[i];
    if (value == 0.F)
    {
      continue;
    }
    const uint32_t key = abs_key(value);
    if ((key & coverage) == prefix)
    {
      atomicAdd(&local[(key >> shift) & mask], 1U);
    }
    else if (track_lower && key < prefix)
    {
      local_lower = max(local_lower, key);
    }
  }
  __syncthreads();
  for (uint32_t bin = threadIdx.x; bin < select_bins; bin += blockDim.x)
  {
    if (local[bin] != 0)
    {
      atomicAdd(&state.histogram[bin], local[bin]);
    }
  }
  if (track_lower)
  {
    const uint32_t block_lower = block_reduce(local_lower, UintMax{});
    if (threadIdx.x == 0 && block_lower != 0)
    {
      atomicMax(&state.lower_max, block_lower);
    }
  }

  __threadfence();
  __shared__ bool last_block;
  if (threadIdx.x == 0)
  {
    last_block = atomicAdd(&state.done, 1U) == stamp_blocks - 1;
  }
  __syncthreads();
  if (!last_block)
  {
    return;
  }

  uint32_t count = state.count;
  if (first_pass)
  {
    uint32_t thread_total = 0;
    for (uint32_t bin = threadIdx.x; bin < select_bins; bin += blockDim.x)
    {
      thread_total += __ldcg(&state.histogram[bin]);
    }
    count = block_reduce(thread_total, cuda::std::plus<>{});
    if (count == 0)
    {
      if (threadIdx.x == 0)
      {
        out_lmad[i_stamp] = 0.F;
      }
      return; // the state stays zero: the later passes skip this stamp
    }
    if (threadIdx.x == 0)
    {
      state.count = count;
    }
  }
  finish_radix_pass(state.histogram, count, shift, mask, true, below, prefix, coverage, state.select, &state.lower_max,
                    &state.done, scale, &out_lmad[i_stamp]);
}

// min and max over every stamp, for gpu_stamp_abs_extreme: stamp_blocks consecutive blocks per stamp, striding over
// its pixels. Each block stores its min and max in its stamp's state; the last block merges them in block order and
// writes `max(|min|, |max|)`.
__global__ void stamp_abs_extreme_kernel(const StampWindows windows, StampExtremeState *states, float *out_extreme)
{
  const unsigned int i_stamp = blockIdx.x / stamp_blocks;
  const unsigned int i_block = blockIdx.x % stamp_blocks;
  StampExtremeState &state = states[i_stamp];
  const Stamp stamp(windows, i_stamp);
  const size_t stamp_pixels = stamp.size();

  float min_value = FLT_MAX;
  float max_value = -FLT_MAX;
  const size_t stride = size_t{stamp_blocks} * blockDim.x;
  for (size_t i = size_t{i_block} * blockDim.x + threadIdx.x; i < stamp_pixels; i += stride)
  {
    const float value = stamp[i];
    min_value = fminf(min_value, value);
    max_value = fmaxf(max_value, value);
  }

  min_value = block_reduce(min_value, FloatMin{});
  max_value = block_reduce(max_value, FloatMax{});

  if (threadIdx.x == 0)
  {
    state.min[i_block] = min_value;
    state.max[i_block] = max_value;
  }
  __threadfence();

  __shared__ bool last_block;
  if (threadIdx.x == 0)
  {
    last_block = atomicAdd(&state.done, 1U) == stamp_blocks - 1;
  }
  __syncthreads();

  if (!last_block)
  {
    return;
  }

  if (threadIdx.x == 0)
  {
    min_value = FLT_MAX;
    max_value = -FLT_MAX;
    for (unsigned int i = 0; i < stamp_blocks; ++i)
    {
      min_value = fminf(min_value, __ldcg(&state.min[i]));
      max_value = fmaxf(max_value, __ldcg(&state.max[i]));
    }
    out_extreme[i_stamp] = fmaxf(fabsf(min_value), fabsf(max_value));
  }
}

__global__ void gather_at_peaks_kernel(const float *plane, const float *noise_map, const float *rms_map,
                                       const float *selfnoise_raw, size_t n_x, const long *peak_y, const long *peak_x,
                                       unsigned int n_islands, float *flux, float *noise_at_peak, float *rms_at_peak,
                                       float *selfnoise_at_peak)
{
  const auto island = global_index();
  if (island >= n_islands)
  {
    return;
  }
  const auto pixel = static_cast<size_t>(peak_y[island]) * n_x + static_cast<size_t>(peak_x[island]);
  flux[island] = plane[pixel];
  noise_at_peak[island] = noise_map[pixel];
  rms_at_peak[island] = rms_map != nullptr ? rms_map[pixel] : 0.F;
  selfnoise_at_peak[island] = selfnoise_raw[pixel];
}

/// Sum, sum of squares and count of the nonzero pixels of a stamp.
struct Moments
{
  double sum;
  double sum_squares;
  uint32_t count;
};

__host__ __device__ Moments operator+(const Moments &a, const Moments &b)
{
  return {a.sum + b.sum, a.sum_squares + b.sum_squares, a.count + b.count};
}

// Moments of the nonzero pixels of every stamp, for gpu_donut_std: stamp_blocks consecutive blocks per stamp, striding
// over its pixels. Each block stores its sum, sum of squares and count in its stamp's state; the last block merges them
// in block order and writes the population standard deviation, or 0 when the stamp has no nonzero pixel.
__global__ void donut_std_kernel(const StampWindows windows, DonutStandardDeviationState *states, float *out_std)
{
  const unsigned int i_stamp = blockIdx.x / stamp_blocks;
  const unsigned int i_block = blockIdx.x % stamp_blocks;
  DonutStandardDeviationState &state = states[i_stamp];
  const Stamp stamp(windows, i_stamp);
  const size_t stamps_pixels = stamp.size();

  Moments moments{0.0, 0.0, 0U};
  const size_t stride = size_t{stamp_blocks} * blockDim.x;
  for (size_t i_pixel = size_t{i_block} * blockDim.x + threadIdx.x; i_pixel < stamps_pixels; i_pixel += stride)
  {
    const auto value = stamp[i_pixel];
    if (value != 0.F)
    {
      moments.sum += value;
      moments.sum_squares += value * value;
      ++moments.count;
    }
  }

  const Moments total = block_reduce(moments, cuda::std::plus<>{});

  if (threadIdx.x == 0)
  {
    state.sum[i_block] = total.sum;
    state.sum_squares[i_block] = total.sum_squares;
    state.count[i_block] = total.count;
  }
  __threadfence();

  __shared__ bool last_block;
  if (threadIdx.x == 0)
  {
    last_block = atomicAdd(&state.done, 1U) == stamp_blocks - 1;
  }
  __syncthreads();

  if (!last_block)
  {
    return;
  }

  if (threadIdx.x == 0)
  {
    double sum = 0.0;
    double sum_square = 0.0;
    uint32_t count = 0U;
    for (unsigned int i = 0; i < stamp_blocks; ++i)
    {
      sum += __ldcg(&state.sum[i]);
      sum_square += __ldcg(&state.sum_squares[i]);
      count += __ldcg(&state.count[i]);
    }

    if (count == 0)
    {
      out_std[i_stamp] = 0.F;
      return;
    }

    const auto mean = sum / static_cast<float>(count);
    const auto mean_square = sum_square / static_cast<float>(count);
    const auto var = mean_square - mean * mean;
    out_std[i_stamp] = sqrtf(static_cast<float>(var));
  }
}

__global__ void stamp_beam_chi2_kernel(const StampWindows windows, const float *kernel, const float kernel_sum,
                                       const float *stamp_extreme, float inv_max_to_std_ratio, float *out_chi2)
{
  const auto i_stamp = blockIdx.x;
  const float adjacent_noise = stamp_extreme[i_stamp] * inv_max_to_std_ratio;

  const Stamp stamp(windows, i_stamp);
  const size_t stamp_pixels = stamp.size();

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
    out_chi2[i_stamp] = (count > 0 && adjacent_noise != 0) ? resid2 / (adjacent_noise * adjacent_noise * count) : 0.F;
  }
}
} // namespace

void gpu_donut_median_abs(const StampWindows &stamps, unsigned int n_islands, float scale, float *out_lmad,
                          IslandScratch &scratch, cudaStream_t stream)
{
  assert(n_islands > 0);
  assert(scratch.donut_median.size() >= n_islands);
  // One memset per call: the passes clear each stamp's histogram and block counter behind them.
  CHECK_GPU(cudaMemsetAsync(scratch.donut_median.data(), 0, n_islands * sizeof(DonutMedianState), stream));
  for (int pass = 0; pass < 3; ++pass)
  {
    launch(donut_median_abs_kernel, n_islands * stamp_blocks, block_size, stream, stamps, select_shifts[pass],
           select_masks[pass], scale, scratch.donut_median.data(), out_lmad);
  }
}

void gpu_donut_std(const StampWindows &stamps, unsigned int n_islands, float *out_std, IslandScratch &scratch,
                   cudaStream_t stream)
{
  assert(n_islands > 0);
  assert(scratch.donut_standard_deviation.size() >= n_islands);
  // Clears the block counters; every block overwrites its own partials.
  CHECK_GPU(cudaMemsetAsync(scratch.donut_standard_deviation.data(), 0,
                            n_islands * sizeof(DonutStandardDeviationState), stream));
  launch(donut_std_kernel, n_islands * stamp_blocks, block_size, stream, stamps,
         scratch.donut_standard_deviation.data(), out_std);
}

void gpu_stamp_abs_extreme(const StampWindows &stamps, unsigned int n_islands, float *out_extreme,
                           IslandScratch &scratch, cudaStream_t stream)
{
  assert(n_islands > 0);
  assert(scratch.stamp_extreme.size() >= n_islands);
  // Clears the block counters; every block overwrites its own partials.
  CHECK_GPU(cudaMemsetAsync(scratch.stamp_extreme.data(), 0, n_islands * sizeof(StampExtremeState), stream));
  launch(stamp_abs_extreme_kernel, n_islands * stamp_blocks, block_size, stream, stamps, scratch.stamp_extreme.data(),
         out_extreme);
}

void gpu_stamp_beam_chi2(const StampWindows &stamps, const float *kernel, const float kernel_sum,
                         unsigned int n_islands, const float *stamp_extreme, float inv_max_to_std_ratio,
                         float *out_chi2, cudaStream_t stream)
{
  launch_per_island(stamp_beam_chi2_kernel, n_islands, stream, stamps, kernel, kernel_sum, stamp_extreme,
                    inv_max_to_std_ratio, out_chi2);
}

void gpu_gather_at_peaks(const float *plane, const float *noise_map, const float *rms_map, const float *selfnoise_raw,
                         size_t n_x, const long *peak_y, const long *peak_x, unsigned int n_islands, float *flux,
                         float *noise_at_peak, float *rms_at_peak, float *selfnoise_at_peak, cudaStream_t stream)
{
  assert(n_islands > 0);
  launch_per_element(gather_at_peaks_kernel, n_islands, stream, plane, noise_map, rms_map, selfnoise_raw, n_x, peak_y,
                     peak_x, n_islands, flux, noise_at_peak, rms_at_peak, selfnoise_at_peak);
}
} // namespace breifast::gpu
