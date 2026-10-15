#include "snr_filters.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tracy/Tracy.hpp>
#include <xsimd/xsimd.hpp>

namespace breifast::plane_lot::detail
{

SnrFilterSummary snr_and_filters(std::span<const float> plane, std::span<const float> noise_map,
                                 std::span<const float> snr_cutoff, std::span<const float> flux_cutoff,
                                 const SnrFilterParams &params, std::span<float> out_snr, std::span<bool> image_mask,
                                 std::array<std::span<long>, 3> histograms)
{
  ZoneScopedN("compute_snr_and_apply_filters");
  const size_t n_pixels = plane.size();
  if (noise_map.size() != n_pixels || out_snr.size() != n_pixels || image_mask.size() != n_pixels)
  {
    throw std::runtime_error("snr_and_filters: noise_map, out_snr or image_mask size mismatch with plane");
  }
  if (params.max_sigma_bin <= 0)
  {
    throw std::runtime_error("snr_and_filters: max_sigma_bin must be positive");
  }
  for (const auto &histogram : histograms)
  {
    if (histogram.size() != static_cast<size_t>(2 * params.max_sigma_bin))
    {
      throw std::runtime_error("snr_and_filters: histogram size mismatch with 2 * max_sigma_bin");
    }
  }
  for (const auto &cutoff : {snr_cutoff, flux_cutoff})
  {
    if (cutoff.empty())
    {
      continue;
    }
    if (params.band_start < 0 || params.band_start >= params.band_stop)
    {
      throw std::runtime_error("snr_and_filters: invalid band");
    }
    if (cutoff.size() < static_cast<size_t>(params.band_stop) * n_pixels)
    {
      throw std::runtime_error("snr_and_filters: cutoff cube has fewer channels than the band needs");
    }
  }
  for (const auto &histogram : histograms)
  {
    std::fill(histogram.begin(), histogram.end(), 0L);
  }

  float local_min_sigma = std::numeric_limits<float>::infinity();
  float local_max_sigma = std::numeric_limits<float>::lowest();

  const float *const plane_ptr = plane.data();
  const float *const noise_map_ptr = noise_map.data();
  float *const snr_ptr = out_snr.data();
  bool *const image_mask_ptr = image_mask.data();
  const bool snr_cutoff_enabled = !snr_cutoff.empty();
  const float *const snr_cutoff_ptr = snr_cutoff.data();
  const bool flux_cutoff_enabled = !flux_cutoff.empty();
  const float *const flux_cutoff_ptr = flux_cutoff.data();

#if SCALAR
  const size_t first_scalar_pixel = 0;
#else
  const xsimd::batch_bool<float> snr_cutoff_disabled(!snr_cutoff_enabled);
  const xsimd::batch_bool<float> flux_cutoff_disabled(!flux_cutoff_enabled);
  const xsimd::batch_bool<float> noise_rejection_disabled(params.reject_noise_threshold == 0.F);
  const xsimd::batch<float> noise_rejection_threshold(params.median_noise * params.reject_noise_threshold);

  constexpr size_t batch_size = xsimd::batch<float>::size;
  const size_t n_vector_pixels = n_pixels - (n_pixels) % batch_size;
  xsimd::batch<float> min_sigma_vec(local_min_sigma);
  xsimd::batch<float> max_sigma_vec(local_max_sigma);
  for (size_t i_pixel = 0; i_pixel < n_vector_pixels; i_pixel += batch_size)
  {
    const auto noise_vec = xsimd::load_unaligned(noise_map_ptr + i_pixel);
    const auto valid = noise_vec != xsimd::batch<float>(0.F) && xsimd::isfinite(noise_vec);
    const auto plane_val = xsimd::load_unaligned(plane_ptr + i_pixel);
    auto snr = xsimd::select(valid, plane_val / noise_vec, xsimd::batch<float>(0.F));
    snr.store_unaligned(snr_ptr + i_pixel);

    auto image_mask_vec = xsimd::batch_bool<float>::load_unaligned(image_mask_ptr + i_pixel);
    image_mask_vec &= valid;
    image_mask_vec.store_unaligned(image_mask_ptr + i_pixel);

    const auto mask0 = image_mask_vec;

    xsimd::batch<float, xsimd::default_arch> snr_cutoff_threshold_vec{0.F};
    if (snr_cutoff_enabled)
    {
      for (auto i_channel = params.band_start; i_channel < params.band_stop; ++i_channel)
      {
        const auto snr_cutoff_vec = xsimd::load_unaligned(snr_cutoff_ptr + i_channel * n_pixels + i_pixel);
        snr_cutoff_threshold_vec += snr_cutoff_vec;
      }
      snr_cutoff_threshold_vec *= params.inv_band_size;
    }

    xsimd::batch<float, xsimd::default_arch> flux_cutoff_threshold_vec{0.F};
    if (flux_cutoff_enabled)
    {
      for (auto i_channel = params.band_start; i_channel < params.band_stop; ++i_channel)
      {
        const auto flux_cutoff_vec = xsimd::load_unaligned(flux_cutoff_ptr + i_channel * n_pixels + i_pixel);
        flux_cutoff_threshold_vec += flux_cutoff_vec;
      }
      flux_cutoff_threshold_vec *= params.inv_band_size;
    }

    const auto mask1 = mask0 && (snr_cutoff_disabled || xsimd::abs(snr) >= snr_cutoff_threshold_vec);
    const auto mask2 = mask1 && (flux_cutoff_disabled || xsimd::abs(plane_val) >= flux_cutoff_threshold_vec);
    const auto mask3 = mask2 && (noise_rejection_disabled || noise_vec <= noise_rejection_threshold);

    mask3.store_unaligned(image_mask_ptr + i_pixel);

    auto bin_index = xsimd::floor(snr + xsimd::batch<float>(params.max_sigma_bin));
    bin_index -= xsimd::select(bin_index == xsimd::batch<float>(2 * params.max_sigma_bin), xsimd::batch<float>(1.F),
                               xsimd::batch<float>(0.F));

    const auto in_range = bin_index >= xsimd::batch<float>(0.F) &&
                          bin_index < xsimd::batch<float>(static_cast<float>(2 * params.max_sigma_bin));
    alignas(64) int32_t bins[xsimd::batch<float>::size];
    xsimd::batch_cast<int32_t>(bin_index).store_aligned(bins);
    const auto add_to_histogram = [&](std::span<long> histogram, const xsimd::batch_bool<float> &mask)
    {
      for (auto bits = (mask && in_range).mask(); bits != 0; bits &= bits - 1)
      {
        histogram[bins[std::countr_zero(bits)]]++;
      }
    };
    if (params.histogram_detail > 1)
    {
      add_to_histogram(histograms[0], mask0);
      add_to_histogram(histograms[1], mask1);
    }
    if (params.histogram_detail > 0)
    {
      add_to_histogram(histograms[2], mask2);
    }

    min_sigma_vec = xsimd::min(min_sigma_vec, xsimd::select(mask3, snr, min_sigma_vec));
    max_sigma_vec = xsimd::max(max_sigma_vec, xsimd::select(mask3, snr, max_sigma_vec));

    const auto thresholded_abs_snr = xsimd::select(mask3, xsimd::abs(snr), xsimd::batch<float>(0.F));
    const auto mask = thresholded_abs_snr >= xsimd::batch<float>(params.threshold);
    mask.store_unaligned(image_mask_ptr + i_pixel);
  }
  local_min_sigma = xsimd::reduce_min(min_sigma_vec);
  local_max_sigma = xsimd::reduce_max(max_sigma_vec);
  const size_t first_scalar_pixel = n_vector_pixels;
#endif

  // Epilog of the xsimd loop: the pixels past the last whole batch.
  for (size_t i_pixel = first_scalar_pixel; i_pixel < n_pixels; ++i_pixel)
  {
    const auto noise = noise_map_ptr[i_pixel];
    const bool valid = noise != 0 && std::isfinite(noise);
    snr_ptr[i_pixel] = valid ? plane_ptr[i_pixel] / noise : 0;
    image_mask_ptr[i_pixel] &= valid;

    const auto mask0 = image_mask_ptr[i_pixel];

    float snr_cutoff_threshold = 0.F;
    if (snr_cutoff_enabled)
    {
      for (auto i_channel = params.band_start; i_channel < params.band_stop; ++i_channel)
      {
        snr_cutoff_threshold += snr_cutoff_ptr[i_channel * n_pixels + i_pixel];
      }
      snr_cutoff_threshold *= params.inv_band_size;
    }

    float flux_cutoff_threshold = 0.F;
    if (flux_cutoff_enabled)
    {
      for (auto i_channel = params.band_start; i_channel < params.band_stop; ++i_channel)
      {
        flux_cutoff_threshold += flux_cutoff_ptr[i_channel * n_pixels + i_pixel];
      }
      flux_cutoff_threshold *= params.inv_band_size;
    }

    const auto mask1 = mask0 && (!snr_cutoff_enabled || std::abs(snr_ptr[i_pixel]) >= snr_cutoff_threshold);
    const auto mask2 = mask1 && (!flux_cutoff_enabled || std::abs(plane_ptr[i_pixel]) >= flux_cutoff_threshold);
    const auto mask3 = mask2 && (params.reject_noise_threshold == 0.F ||
                                 noise_map_ptr[i_pixel] <= params.median_noise * params.reject_noise_threshold);

    image_mask_ptr[i_pixel] = mask3;

    long bin_index = std::floor(snr_ptr[i_pixel] + params.max_sigma_bin);
    if (bin_index == 2 * params.max_sigma_bin)
    {
      bin_index = 2 * params.max_sigma_bin - 1;
    }
    if (0 <= bin_index && bin_index < 2 * params.max_sigma_bin)
    {
      if (mask0 && params.histogram_detail > 1)
      {
        histograms[0][bin_index]++;
      }
      if (mask1 && params.histogram_detail > 1)
      {
        histograms[1][bin_index]++;
      }
      if (mask2 && params.histogram_detail > 0)
      {
        histograms[2][bin_index]++;
      }
    }

    if (mask3)
    {
      const float value = snr_ptr[i_pixel];
      local_min_sigma = std::min(local_min_sigma, value);
      local_max_sigma = std::max(local_max_sigma, value);
    }

    const float thresholded_abs_snr = mask3 ? std::abs(snr_ptr[i_pixel]) : 0.F;
    image_mask_ptr[i_pixel] = thresholded_abs_snr >= params.threshold;
  }
  return {local_min_sigma, local_max_sigma};
}

} // namespace breifast::plane_lot::detail
