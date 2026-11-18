#include <algorithm>
#include <cassert>
#include <cuda/std/limits>

#include "gpu_kernels.hpp"

#include "gpu_device_utils.cuh"
#include "common/snr_filter_params.hpp"

namespace breifast::gpu
{
namespace
{
__global__ void min_max_init(float *out_min_sigma, float *out_max_sigma)
{
  *out_min_sigma = __int_as_float(0x7f800000);  // +inf
  *out_max_sigma = -__int_as_float(0x7f7fffff); // -FLT_MAX (lowest)
}

/// Upper bound on the grid of snr_and_filters: each block pays a fixed cost (zeroing and flushing its shared
/// histograms, two block reductions, two global compare-and-swaps), so the threads loop over the pixels instead.
constexpr unsigned int snr_max_blocks = 1024;

__global__ void snr_and_filters(const float *plane, const float *noise_map, const float *snr_cutoff,
                                const float *flux_cutoff, const float *median_noise, const size_t n_pixels,
                                const common::SnrFilterParams params, float *out_snr, bool *image_mask,
                                unsigned long long *histogram0, unsigned long long *histogram1,
                                unsigned long long *histogram2, float *out_min_sigma, float *out_max_sigma)
{
  constexpr float inf = cuda::std::numeric_limits<float>::infinity();
  const bool snr_cutoff_enabled = snr_cutoff != nullptr;
  const bool flux_cutoff_enabled = flux_cutoff != nullptr;
  const bool noise_rejection_disabled = params.reject_noise_threshold == 0.F;
  const float noise_rejection_threshold = median_noise[0] * params.reject_noise_threshold;

  const auto n_bins = static_cast<size_t>(2 * params.max_sigma_bin);
  extern __shared__ unsigned long long shared_histograms[];
  auto *const shared_histogram0 = shared_histograms;
  auto *const shared_histogram1 = shared_histograms + n_bins;
  auto *const shared_histogram2 = shared_histograms + 2 * n_bins;
  for (size_t i_bin = threadIdx.x; i_bin < 3 * n_bins; i_bin += blockDim.x)
  {
    shared_histograms[i_bin] = 0;
  }
  __syncthreads();

  float local_min = inf;
  float local_max = -inf;
  const auto process = [&](const float plane_val, const float noise, const bool mask_in, const float snr_threshold,
                           const float flux_threshold, float &snr) -> bool
  {
    const bool valid = noise != 0.F && isfinite(noise);
    snr = valid ? plane_val / noise : 0.F;

    const bool mask0 = mask_in && valid;
    const bool mask1 = mask0 && (!snr_cutoff_enabled || fabsf(snr) >= snr_threshold);
    const bool mask2 = mask1 && (!flux_cutoff_enabled || fabsf(plane_val) >= flux_threshold);
    const bool mask3 = mask2 && (noise_rejection_disabled || noise <= noise_rejection_threshold);

    float bin_index = floorf(snr + params.max_sigma_bin);
    bin_index -= (bin_index == 2 * params.max_sigma_bin ? 1.F : 0.F);
    if (0 <= bin_index && bin_index < n_bins)
    {
      const auto bin = static_cast<size_t>(bin_index);
      if (mask0 && params.histogram_detail > 1)
      {
        atomicAdd(&shared_histogram0[bin], 1ULL);
      }
      if (mask1 && params.histogram_detail > 1)
      {
        atomicAdd(&shared_histogram1[bin], 1ULL);
      }
      if (mask2 && params.histogram_detail > 0)
      {
        atomicAdd(&shared_histogram2[bin], 1ULL);
      }
    }

    if (mask3)
    {
      local_min = fminf(local_min, snr);
      local_max = fmaxf(local_max, snr);
    }
    return (mask3 ? fabsf(snr) : 0.F) >= params.threshold;
  };

  const auto band_mean = [&](const float *cutoff, const size_t i_pixel)
  {
    float sum = 0.F;
    for (auto i_channel = params.band_start; i_channel < params.band_stop; ++i_channel)
    {
      sum += cutoff[i_channel * n_pixels + i_pixel];
    }
    return sum * params.inv_band_size;
  };

  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  const size_t n4 = n_pixels % 4 == 0 ? n_pixels / 4 : 0;
  for (size_t i4 = global_index(); i4 < n4; i4 += stride)
  {
    const float4 value4 = reinterpret_cast<const float4 *>(plane)[i4];
    const float4 noise4 = reinterpret_cast<const float4 *>(noise_map)[i4];
    const uchar4 mask4 = reinterpret_cast<const uchar4 *>(image_mask)[i4];

    float4 snr_cutoff_sum = {0.F, 0.F, 0.F, 0.F};
    float4 flx_cutoff_sum = {0.F, 0.F, 0.F, 0.F};

    for (auto i_channel = params.band_start; snr_cutoff_enabled && i_channel < params.band_stop; ++i_channel)
    {
      const float4 snr_cutoff_value = reinterpret_cast<const float4 *>(snr_cutoff + i_channel * n_pixels)[i4];
      snr_cutoff_sum.x += snr_cutoff_value.x;
      snr_cutoff_sum.y += snr_cutoff_value.y;
      snr_cutoff_sum.z += snr_cutoff_value.z;
      snr_cutoff_sum.w += snr_cutoff_value.w;
    }

    for (auto i_channel = params.band_start; flux_cutoff_enabled && i_channel < params.band_stop; ++i_channel)
    {
      const float4 flx_cutoff_value = reinterpret_cast<const float4 *>(flux_cutoff + i_channel * n_pixels)[i4];
      flx_cutoff_sum.x += flx_cutoff_value.x;
      flx_cutoff_sum.y += flx_cutoff_value.y;
      flx_cutoff_sum.z += flx_cutoff_value.z;
      flx_cutoff_sum.w += flx_cutoff_value.w;
    }

    float4 snr4;
    uchar4 out4;
    out4.x = process(value4.x, noise4.x, mask4.x != 0, snr_cutoff_sum.x * params.inv_band_size,
                     flx_cutoff_sum.x * params.inv_band_size, snr4.x);
    out4.y = process(value4.y, noise4.y, mask4.y != 0, snr_cutoff_sum.y * params.inv_band_size,
                     flx_cutoff_sum.y * params.inv_band_size, snr4.y);
    out4.z = process(value4.z, noise4.z, mask4.z != 0, snr_cutoff_sum.z * params.inv_band_size,
                     flx_cutoff_sum.z * params.inv_band_size, snr4.z);
    out4.w = process(value4.w, noise4.w, mask4.w != 0, snr_cutoff_sum.w * params.inv_band_size,
                     flx_cutoff_sum.w * params.inv_band_size, snr4.w);

    reinterpret_cast<uchar4 *>(image_mask)[i4] = out4;
    reinterpret_cast<float4 *>(out_snr)[i4] = snr4;
  }

  for (size_t i_pixel = 4 * n4 + global_index(); i_pixel < n_pixels; i_pixel += stride)
  {
    float snr;
    image_mask[i_pixel] = process(plane[i_pixel], noise_map[i_pixel], image_mask[i_pixel],
                                  snr_cutoff_enabled ? band_mean(snr_cutoff, i_pixel) : 0.0f,
                                  flux_cutoff_enabled ? band_mean(flux_cutoff, i_pixel) : 0.0f, snr);
    out_snr[i_pixel] = snr;
  }

  __syncthreads();
  for (size_t i_bin = threadIdx.x; i_bin < n_bins; i_bin += blockDim.x)
  {
    if (params.histogram_detail > 1 && shared_histogram0[i_bin] != 0)
    {
      atomicAdd(histogram0 + i_bin, shared_histogram0[i_bin]);
    }
    if (params.histogram_detail > 1 && shared_histogram1[i_bin] != 0)
    {
      atomicAdd(histogram1 + i_bin, shared_histogram1[i_bin]);
    }
    if (params.histogram_detail > 0 && shared_histogram2[i_bin] != 0)
    {
      atomicAdd(histogram2 + i_bin, shared_histogram2[i_bin]);
    }
  }

  const float block_min = block_reduce(local_min, FloatMin{});
  const float block_max = block_reduce(local_max, FloatMax{});
  if (threadIdx.x == 0 && block_min <= block_max)
  {
    atomic_update(out_min_sigma, block_min, FloatMin{});
    atomic_update(out_max_sigma, block_max, FloatMax{});
  }
}
} // namespace

void gpu_snr_and_filters(const float *plane, const float *noise_map, const float *snr_cutoff, const float *flux_cutoff,
                         const float *median_noise, size_t n_pixels, const common::SnrFilterParams &params,
                         float *out_snr, bool *image_mask, unsigned long long *histogram0,
                         unsigned long long *histogram1, unsigned long long *histogram2, float *out_min_sigma,
                         float *out_max_sigma, cudaStream_t stream)
{
  assert(n_pixels > 0);
  assert(params.max_sigma_bin > 0);
  assert(median_noise != nullptr);
  assert(params.histogram_detail <= 1 || (histogram0 != nullptr && histogram1 != nullptr));
  assert(params.histogram_detail <= 0 || histogram2 != nullptr);

  if (histogram0 != nullptr)
  {
    CHECK_GPU(cudaMemsetAsync(histogram0, 0, sizeof(unsigned long long) * 2 * params.max_sigma_bin, stream));
  }
  if (histogram1 != nullptr)
  {
    CHECK_GPU(cudaMemsetAsync(histogram1, 0, sizeof(unsigned long long) * 2 * params.max_sigma_bin, stream));
  }
  if (histogram2 != nullptr)
  {
    CHECK_GPU(cudaMemsetAsync(histogram2, 0, sizeof(unsigned long long) * 2 * params.max_sigma_bin, stream));
  }
  launch(min_max_init, 1, 1, stream, out_min_sigma, out_max_sigma);
  // Dynamic shared memory: the block's three histograms, 2 * max_sigma_bin bins each.
  const auto shared_bytes = 3 * 2 * static_cast<std::size_t>(params.max_sigma_bin) * sizeof(unsigned long long);
  const auto grid =
      static_cast<unsigned int>(std::min<std::size_t>((n_pixels + block_size - 1) / block_size, snr_max_blocks));
  snr_and_filters<<<grid, block_size, shared_bytes, stream>>>(plane, noise_map, snr_cutoff, flux_cutoff, median_noise,
                                                              n_pixels, params, out_snr, image_mask, histogram0,
                                                              histogram1, histogram2, out_min_sigma, out_max_sigma);
  CHECK_GPU(cudaGetLastError());
}

} // namespace breifast::gpu