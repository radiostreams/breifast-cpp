#include <cstdint>

#include "gpu_pipeline.hpp"
#include "gpu_runtime.hpp"
#include "radix_keys.cuh"

namespace breifast::gpu
{
namespace
{
constexpr size_t n_digits = 4;
constexpr size_t histogram_bins = 1 << 8;

// One byte per pass, most significant first, of the keys whose resolved high bytes equal the prefix. Counts into a
// device-wide 256-bin histogram.
__global__ void radix_histogram_kernel(const float *values, size_t n, uint32_t *histogram, int shift, bool abs_mode,
                                       const uint32_t *state)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    const uint32_t key = abs_mode ? abs_key(values[idx]) : signed_key(values[idx]);
    if ((key & state[2]) == state[1])
    {
      atomicAdd(&histogram[(key >> shift) & 0xFFU], 1U);
    }
  }
}

// Cumulative count over the bins, stopping where rank `k` lands. state[0] is the count below the current prefix, which
// this advances; state[1] and state[2] extend the prefix and the coverage with the chosen byte.
__global__ void radix_scan_kernel(const uint32_t *histogram, size_t rank, int shift, uint32_t *state)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    size_t cumulative = state[0];
    uint32_t bin = 0;
    for (; bin < histogram_bins; ++bin)
    {
      if (cumulative + histogram[bin] > rank)
      {
        break;
      }
      cumulative += histogram[bin];
    }
    state[0] = static_cast<uint32_t>(cumulative);
    state[1] |= bin << shift;
    state[2] |= 0xFFU << shift;
  }
}

// Even n only: count the keys equal to the resolved rank-k key (aux[0]) and the largest key strictly below it (aux[1]),
// to resolve rank k-1.
__global__ void radix_refine_kernel(const float *values, size_t n, bool abs_mode, const uint32_t *state, uint32_t *aux)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    const uint32_t key = abs_mode ? abs_key(values[idx]) : signed_key(values[idx]);
    if (key == state[1])
    {
      atomicAdd(&aux[0], 1U);
    }
    else if (key < state[1])
    {
      atomicMax(&aux[1], key);
    }
  }
}

// state[1] is the full rank-k key. Even n averages it with rank k-1: the same key when it repeats, else the largest key
// below it.
__global__ void radix_finalize_kernel(const uint32_t *state, const uint32_t *aux, bool abs_mode, size_t n,
                                      float *out_median)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    const uint32_t hi_key = state[1];
    const float hi = abs_mode ? abs_value(hi_key) : signed_value(hi_key);
    if (n % 2 == 1)
    {
      *out_median = hi;
      return;
    }
    const uint32_t lo_key = aux[0] >= 2 ? hi_key : aux[1];
    const float lo = abs_mode ? abs_value(lo_key) : signed_value(lo_key);
    *out_median = (lo + hi) / 2.F;
  }
}

__global__ void write_nan_kernel(float *out)
{
  *out = __int_as_float(0x7FC00000);
}

void gpu_median_impl(const float *values, size_t n, float *out_median, bool abs_mode, GpuStream stream)
{
  if (n == 0)
  {
    write_nan_kernel<<<1, 1, 0, stream>>>(out_median);
    CHECK_LAST_GPU_ERROR();
    return;
  }

  constexpr size_t threads = 256;
  const dim3 block_size = {threads};
  const dim3 grid_size = {static_cast<unsigned int>((n + threads - 1) / threads)};

  constexpr size_t scratch_words = histogram_bins + 5; // 256 bins, then state[3] and aux[2]
  uint32_t *scratch = nullptr;
  gpuMallocAsync(&scratch, scratch_words * sizeof(uint32_t), stream);
  gpuMemsetAsync(scratch, 0, scratch_words * sizeof(uint32_t), stream);

  uint32_t *histogram = scratch;
  uint32_t *state = scratch + histogram_bins;   // below, prefix, coverage
  uint32_t *aux = scratch + histogram_bins + 3; // count_eq, max_below

  const int shifts[n_digits] = {24, 16, 8, 0};
  for (int shift : shifts)
  {
    gpuMemsetAsync(histogram, 0, histogram_bins * sizeof(uint32_t), stream);
    radix_histogram_kernel<<<grid_size, block_size, 0, stream>>>(values, n, histogram, shift, abs_mode, state);
    CHECK_LAST_GPU_ERROR();
    radix_scan_kernel<<<1, 1, 0, stream>>>(histogram, n / 2, shift, state);
    CHECK_LAST_GPU_ERROR();
  }

  if (n % 2 == 0)
  {
    radix_refine_kernel<<<grid_size, block_size, 0, stream>>>(values, n, abs_mode, state, aux);
    CHECK_LAST_GPU_ERROR();
  }

  radix_finalize_kernel<<<1, 1, 0, stream>>>(state, aux, abs_mode, n, out_median);
  CHECK_LAST_GPU_ERROR();

  gpuFreeAsync(scratch, stream);
}
} // namespace

void gpu_median(const float *values, size_t n, float *out_median, GpuStream stream)
{
  gpu_median_impl(values, n, out_median, false, stream);
}

void gpu_median_abs(const float *values, size_t n, float *out_median, GpuStream stream)
{
  gpu_median_impl(values, n, out_median, true, stream);
}
} // namespace breifast::gpu
