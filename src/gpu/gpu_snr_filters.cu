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

  // The block's histograms, flushed to the global ones once at the end.
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
  for (size_t i_pixel = global_index(); i_pixel < n_pixels; i_pixel += size_t{gridDim.x} * blockDim.x)
  {
    const float noise = noise_map[i_pixel];
    const bool valid = noise != 0.0f && isfinite(noise);
    const float plane_val = plane[i_pixel];
    const float snr = valid ? plane_val / noise : 0.0f;
    const bool mask0 = image_mask[i_pixel] && valid;

    float snr_cutoff_threshold = 0.0f;
    if (snr_cutoff_enabled)
    {
      for (auto i_channel = params.band_start; i_channel < params.band_stop; ++i_channel)
      {
        snr_cutoff_threshold += snr_cutoff[i_channel * n_pixels + i_pixel];
      }
      snr_cutoff_threshold *= params.inv_band_size;
    }

    float flux_cutoff_threshold = 0.0f;
    if (flux_cutoff_enabled)
    {
      for (auto i_channel = params.band_start; i_channel < params.band_stop; ++i_channel)
      {
        flux_cutoff_threshold += flux_cutoff[i_channel * n_pixels + i_pixel];
      }
      flux_cutoff_threshold *= params.inv_band_size;
    }

    const bool mask1 = mask0 && (!snr_cutoff_enabled || fabsf(snr) >= snr_cutoff_threshold);
    const bool mask2 = mask1 && (!flux_cutoff_enabled || fabsf(plane_val) >= flux_cutoff_threshold);
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

    // The intermediate masks were only ever overwritten by this one: a single write.
    image_mask[i_pixel] = (mask3 ? fabsf(snr) : 0.0f) >= params.threshold;
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
  const auto grid = static_cast<unsigned int>(
      std::min<std::size_t>((n_pixels + block_size - 1) / block_size, snr_max_blocks));
  snr_and_filters<<<grid, block_size, shared_bytes, stream>>>(
      plane, noise_map, snr_cutoff, flux_cutoff, median_noise, n_pixels, params, out_snr, image_mask, histogram0,
      histogram1, histogram2, out_min_sigma, out_max_sigma);
  CHECK_GPU(cudaGetLastError());
}

} // namespace breifast::gpu