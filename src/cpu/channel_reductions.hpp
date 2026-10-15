#pragma once
#include <cstddef>
#include <span>

namespace breifast::plane_lot::detail
{

/**
 * @brief OR of the accept masks of the channels `[band_start, band_stop)`, and whether any pixel ends up true.
 *
 * Copies the first channel into `out_mask` and ORs the others, so it writes `out_mask` completely.
 * @param accept_mask `[n_channels, n_pixels]` bools, channel major.
 * @param out_mask `[n_pixels]` bools.
 * @return True when any pixel of `out_mask` is true.
 */
bool or_accept_masks(std::span<const bool> accept_mask, const size_t n_pixels, const long band_start,
                     const long band_stop, std::span<bool> out_mask);

/**
 * @brief Weighted mean of the band's channels, minus their mean image when one is given:
 *        `out[p] = sum_c cube[c, p] * weights[c] * inv_weights_sum - sum_c mean_image[c, p] * inv_band_size`.
 *
 * The first channel assigns and the others add, and the mean term follows each channel's value, so the rounding is the
 * same as a per-pixel loop over the channels. Writes `out_plane` completely.
 * @param cube `[n_channels, n_pixels]` values, channel major.
 * @param weights One weight per channel, indexed by the absolute channel.
 * @param mean_image `[n_channels, n_pixels]` values, or empty for no subtraction.
 */
void weighted_band_image(std::span<const float> cube, std::span<const float> weights, std::span<const float> mean_image,
                         const size_t n_pixels, const long band_start, const long band_stop,
                         const float inv_weights_sum, const float inv_band_size, std::span<float> out_plane);

/**
 * @brief Per-pixel sums of the second, third and fourth powers over the valid channels:
 *        `out_sum_k[p] = sum over channels c with nonzero_channels[c] of cube[c, p]^k`, k = 2, 3, 4.
 *
 * Channels flagged false are skipped without being read. The powers are built as `x * x`, `x^2 * x`, `x^3 * x` in float
 * and added in channel order. Writes the three sums completely.
 * @param cube `[n_channels, n_pixels]` values, channel major, `n_channels = nonzero_channels.size()`.
 */
void spectral_variance_moments(std::span<const float> cube, std::span<const bool> nonzero_channels,
                               const size_t n_pixels, std::span<float> out_sum2, std::span<float> out_sum3,
                               std::span<float> out_sum4);

} // namespace breifast::plane_lot::detail
