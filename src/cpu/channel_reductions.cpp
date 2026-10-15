#include "channel_reductions.hpp"

#include <algorithm>
#include <stdexcept>
#include <tracy/Tracy.hpp>
#include <type_traits>

#include "scalar_reductions.hpp"

namespace breifast::plane_lot::detail
{

bool or_accept_masks(std::span<const bool> accept_mask, const size_t n_pixels, const long band_start,
                     const long band_stop, std::span<bool> out_mask)
{
  ZoneScopedN("mask_accumulate");
  if (band_start < 0 || band_start >= band_stop)
  {
    throw std::runtime_error("or_accept_masks: invalid band");
  }
  if (accept_mask.size() < static_cast<size_t>(band_stop) * n_pixels)
  {
    throw std::runtime_error("or_accept_masks: accept_mask has fewer channels than the band needs");
  }
  if (out_mask.size() != n_pixels)
  {
    throw std::runtime_error("or_accept_masks: out_mask size mismatch with n_pixels");
  }

  bool *const __restrict out = out_mask.data();
  std::copy_n(accept_mask.data() + static_cast<size_t>(band_start) * n_pixels, n_pixels, out);
  for (long i_channel = band_start + 1; i_channel < band_stop; ++i_channel)
  {
    const bool *const __restrict channel = accept_mask.data() + static_cast<size_t>(i_channel) * n_pixels;
    for (size_t i_pixel = 0; i_pixel < n_pixels; ++i_pixel)
    {
      out[i_pixel] = out[i_pixel] | channel[i_pixel];
    }
  }
  return any_true(out, n_pixels);
}

void weighted_band_image(std::span<const float> cube, std::span<const float> weights, std::span<const float> mean_image,
                         const size_t n_pixels, const long band_start, const long band_stop,
                         const float inv_weights_sum, const float inv_band_size, std::span<float> out_plane)
{
  ZoneScopedN("band_accumulate");
  if (band_start < 0 || band_start >= band_stop)
  {
    throw std::runtime_error("weighted_band_image: invalid band");
  }
  const size_t n_channels_needed = static_cast<size_t>(band_stop);
  if (cube.size() < n_channels_needed * n_pixels)
  {
    throw std::runtime_error("weighted_band_image: cube has fewer channels than the band needs");
  }
  if (weights.size() < n_channels_needed)
  {
    throw std::runtime_error("weighted_band_image: weights has fewer entries than the band needs");
  }
  const bool subtract_mean = !mean_image.empty();
  if (subtract_mean && mean_image.size() < n_channels_needed * n_pixels)
  {
    throw std::runtime_error("weighted_band_image: mean_image has fewer channels than the band needs");
  }
  if (out_plane.size() != n_pixels)
  {
    throw std::runtime_error("weighted_band_image: out_plane size mismatch with n_pixels");
  }

  float *const __restrict out = out_plane.data();
  const auto accumulate = [&](size_t i_channel, auto first, auto subtract)
  {
    const float *const __restrict cube_c = cube.data() + i_channel * n_pixels;
    const float *const mean_c = subtract ? mean_image.data() + i_channel * n_pixels : nullptr;
    const float weight = weights[i_channel];
    for (size_t i = 0; i < n_pixels; ++i)
    {
      const float value = cube_c[i] * weight * inv_weights_sum;
      if constexpr (first)
      {
        out[i] = value;
      }
      else
      {
        out[i] += value;
      }
      if constexpr (subtract)
      {
        out[i] -= mean_c[i] * inv_band_size;
      }
    }
  };

  const auto accumulate_channel = [&](size_t i_channel, auto first)
  {
    if (subtract_mean)
    {
      accumulate(i_channel, first, std::true_type{});
    }
    else
    {
      accumulate(i_channel, first, std::false_type{});
    }
  };

  accumulate_channel(static_cast<size_t>(band_start), std::true_type{});
  for (long i_channel = band_start + 1; i_channel < band_stop; ++i_channel)
  {
    accumulate_channel(static_cast<size_t>(i_channel), std::false_type{});
  }
}

void spectral_variance_moments(std::span<const float> cube, std::span<const bool> nonzero_channels,
                               const size_t n_pixels, std::span<float> out_sum2, std::span<float> out_sum3,
                               std::span<float> out_sum4)
{
  ZoneScopedN("compute_spectral_variance_stats");
  const size_t n_channels = nonzero_channels.size();
  if (cube.size() != n_channels * n_pixels)
  {
    throw std::runtime_error("spectral_variance_moments: cube size mismatch with nonzero_channels and n_pixels");
  }
  if (out_sum2.size() != n_pixels || out_sum3.size() != n_pixels || out_sum4.size() != n_pixels)
  {
    throw std::runtime_error("spectral_variance_moments: output size mismatch with n_pixels");
  }

  std::fill(out_sum2.begin(), out_sum2.end(), 0.F);
  std::fill(out_sum3.begin(), out_sum3.end(), 0.F);
  std::fill(out_sum4.begin(), out_sum4.end(), 0.F);

  const float *__restrict cube_ptr = cube.data();
  float *__restrict sum_pow2_ptr = out_sum2.data();
  float *__restrict sum_pow3_ptr = out_sum3.data();
  float *__restrict sum_pow4_ptr = out_sum4.data();

  for (size_t i_channel = 0; i_channel < n_channels; ++i_channel)
  {
    const float *__restrict local_cube_ptr = cube_ptr + i_channel * n_pixels;
    if (!nonzero_channels[i_channel])
    {
      continue;
    }
#pragma GCC ivdep
    for (size_t i_pixel = 0; i_pixel < n_pixels; ++i_pixel)
    {
      const auto value = local_cube_ptr[i_pixel];
      const auto value2 = value * value;
      const auto value3 = value2 * value;
      const auto value4 = value3 * value;
      sum_pow2_ptr[i_pixel] += value2;
      sum_pow3_ptr[i_pixel] += value3;
      sum_pow4_ptr[i_pixel] += value4;
    }
  }
}

} // namespace breifast::plane_lot::detail
