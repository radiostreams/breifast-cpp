#include "gpu_band.hpp"
#include "gpu_kernels.hpp"
#include "radix_select.cuh"

namespace breifast::gpu
{
namespace
{
#if 0
// Variant to revisit: one private histogram per thread, so no atomics and no bank conflicts. Disabled because it does
// not fit: radix_bins * block_size counters are 256 KB of shared memory, 48 KB at most (ptxas: "uses too much shared
// data"). Idea to explore: fewer bins per sweep.
__global__ void radix_histogram_private_kernel(const float *values, size_t n, uint32_t *histogram, int shift,
                                               bool abs_mode, const uint32_t *state)
{
  __shared__ uint32_t local[radix_bins * block_size];
  for (size_t bin = 0; bin < radix_bins; ++bin)
  {
    local[block_size * bin + threadIdx.x] = 0;
  }
  /*for (size_t bin = threadIdx.x; bin < radix_bins; bin += blockDim.x)
  {
    local[bin] = 0;
  }*/
  __syncthreads();

  const uint32_t prefix = state[1];
  const uint32_t coverage = state[2];
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = global_index(); idx < n; idx += stride)
  {
    const uint32_t key = abs_mode ? abs_key(values[idx]) : signed_key(values[idx]);
    if ((key & coverage) == prefix)
    {
      const auto bin = (key >> shift) & 0xFFU;
      ++local[bin * block_size + threadIdx.x];
    }
  }
  __syncthreads();

  for (size_t bin = threadIdx.x; bin < radix_bins; bin += blockDim.x)
  {
    for (size_t i = 1; i < block_size; ++i)
    {
      local[bin * block_size] += local[bin * block_size + i];
    }
    if (local[bin * block_size] != 0)
    {
      atomicAdd(&histogram[bin], local[bin * block_size]);
    }
  }
}
#endif

void gpu_median_impl(const float *values, size_t n, float *out_median, bool abs_mode, float scale,
                     MedianScratch &scratch, cudaStream_t stream)
{
  radix_median(RadixValues{values}, values, n, abs_mode, scale, out_median, scratch, stream);
}
} // namespace

void gpu_median(const float *values, size_t n, float *out_median, MedianScratch &scratch, cudaStream_t stream)
{
  gpu_median_impl(values, n, out_median, false, 1.F, scratch, stream);
}

void gpu_median_abs(const float *values, size_t n, float *out_median, float scale, MedianScratch &scratch,
                    cudaStream_t stream)
{
  gpu_median_impl(values, n, out_median, true, scale, scratch, stream);
}
} // namespace breifast::gpu
