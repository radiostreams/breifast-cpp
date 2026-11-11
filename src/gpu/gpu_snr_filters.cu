#include "gpu_pipeline.hpp"

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

__global__ void snr_and_filters(const float *plane, const float *noise_map, const float *snr_cutoff,
                                const float *flux_cutoff, const float *median_noise, const size_t n_pixels,
                                const common::SnrFilterParams params,
                                float *out_snr, bool *image_mask, unsigned long long *histogram0,
                                unsigned long long *histogram1, unsigned long long *histogram2, float *out_min_sigma,
                                float *out_max_sigma)
{
  const auto i_pixel = global_index();
  if (i_pixel < n_pixels)
  {
    const auto noise = noise_map[i_pixel];
    const auto valid = noise != 0.0f && std::isfinite(noise);
    const auto plane_val = plane[i_pixel];
    const auto snr = valid ? plane_val / noise : 0.0f;

    const bool snr_cutoff_enabled = snr_cutoff != nullptr;
    const bool flux_cutoff_enabled = flux_cutoff != nullptr;
    const bool noise_rejection_disabled = params.reject_noise_threshold == 0.F;
    const float noise_rejection_threshold = median_noise[0] * params.reject_noise_threshold;

    image_mask[i_pixel] &= valid;

    const auto mask0 = image_mask[i_pixel];

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

    const auto mask1 = mask0 && (!snr_cutoff_enabled || fabsf(snr) >= snr_cutoff_threshold);
    const auto mask2 = mask1 && (!flux_cutoff_enabled || fabsf(plane_val) >= flux_cutoff_threshold);
    const auto mask3 = mask2 && (noise_rejection_disabled || noise <= noise_rejection_threshold);

    image_mask[i_pixel] = mask3;

    auto bin_index = floorf(snr + params.max_sigma_bin);
    bin_index -= (bin_index == 2 * params.max_sigma_bin ? 1.F : +0.F);

    if (0 <= bin_index && bin_index < 2 * params.max_sigma_bin)
    {
      if (mask0 && params.histogram_detail > 1)
      {
        atomicAdd(&histogram0[static_cast<unsigned long long>(bin_index)], 1ULL);
      }
      if (mask1 && params.histogram_detail > 1)
      {
        atomicAdd(&histogram1[static_cast<unsigned long long>(bin_index)], 1ULL);
      }
      if (mask2 && params.histogram_detail > 0)
      {
        atomicAdd(&histogram2[static_cast<unsigned long long>(bin_index)], 1ULL);
      }
    }

    if (mask3)
    {
      atomic_update(out_min_sigma, snr, FloatMin{});
      atomic_update(out_max_sigma, snr, FloatMax{});
    }

    const float thresholded_abs_snr = mask3 ? fabsf(snr) : 0.0f;
    image_mask[i_pixel] = thresholded_abs_snr >= params.threshold;
    out_snr[i_pixel] = snr;
  }
}
} // namespace

void gpu_snr_and_filters(const float *plane, const float *noise_map, const float *snr_cutoff, const float *flux_cutoff,
                         const float *median_noise, size_t n_pixels, const common::SnrFilterParams &params,
                         float *out_snr, bool *image_mask,
                         unsigned long long *histogram0, unsigned long long *histogram1, unsigned long long *histogram2,
                         float *out_min_sigma, float *out_max_sigma, GpuStream stream)
{
  if (n_pixels == 0)
  {
    throw std::runtime_error("gpu_snr_and_filters: zero pixels");
  }
  if (params.max_sigma_bin <= 0)
  {
    throw std::runtime_error("gpu_snr_and_filters: max_sigma_bin must be positive");
  }
  if (median_noise == nullptr)
  {
    throw std::runtime_error("gpu_snr_and_filters: median_noise must be non-null");
  }

  if (params.histogram_detail > 1)
  {
    if (histogram0 == nullptr || histogram1 == nullptr)
    {
      throw std::runtime_error(
          "gpu_snr_and_filters: histogram0 and histogram1 must be non-null when histogram_detail > 1");
    }
  }
  if (params.histogram_detail > 0)
  {
    if (histogram2 == nullptr)
    {
      throw std::runtime_error("gpu_snr_and_filters: histogram2 must be non-null when histogram_detail > 0");
    }
  }
  if (histogram0 != nullptr)
  {
    gpuMemsetAsync(histogram0, 0, sizeof(unsigned long long) * 2 * params.max_sigma_bin, stream);
  }
  if (histogram1 != nullptr)
  {
    gpuMemsetAsync(histogram1, 0, sizeof(unsigned long long) * 2 * params.max_sigma_bin, stream);
  }
  if (histogram2 != nullptr)
  {
    gpuMemsetAsync(histogram2, 0, sizeof(unsigned long long) * 2 * params.max_sigma_bin, stream);
  }
  launch(min_max_init, 1, 1, stream, out_min_sigma, out_max_sigma);
  launch_per_element(snr_and_filters, n_pixels, stream, plane, noise_map, snr_cutoff, flux_cutoff, median_noise,
                     n_pixels, params, out_snr, image_mask, histogram0, histogram1, histogram2, out_min_sigma,
                     out_max_sigma);
}

} // namespace breifast::gpu