#pragma once
#include <cstddef>
#include <span>

namespace breifast::plane_lot::detail
{

/**
 * @brief Copies the `(2 * radius + 1)^2` window of `plane` centred on `(centre_y, centre_x)` into `out_stamp`, with
 *        zeros wherever the window leaves the plane.
 *
 * The same stamp as slicing a plane padded with `radius` zeros on every side, without building the padded plane. Writes
 * `out_stamp` completely.
 * @param plane `[n_y, n_x]` values, row major.
 * @param out_stamp `[2 * radius + 1, 2 * radius + 1]` values, row major.
 */
void stamp_extract(std::span<const float> plane, size_t n_y, size_t n_x, long centre_y, long centre_x, long radius,
                   std::span<float> out_stamp);

/**
 * @brief Zeros the `(2 * inner_boxsize + 1)^2` block at the centre of a square stamp, in place. A block wider than the
 *        stamp is clipped to it, so `inner_boxsize >= radius` zeros the whole stamp.
 * @param stamp `[2 * radius + 1, 2 * radius + 1]` values, row major.
 */
void stamp_zero_inner_box(std::span<float> stamp, long radius, long inner_boxsize);

/**
 * @brief `max(|min|, |max|)` over the stamp, from a running minimum starting at `FLT_MAX` and a running maximum
 * starting at `lowest()`. NaN pixels are ignored, and a stamp without any other value gives `FLT_MAX`.
 */
float stamp_abs_extreme(std::span<const float> stamp);

/// Spread of the nonzero pixels of a stamp, see stamp_donut_stats().
struct DonutStats
{
  double stddev; ///< xt::stddev of the nonzero pixels (a double), 0 when there are none
  float lmad;    ///< 1.4826 * xt::median of their absolute values, 0 when there are none
};

/**
 * @brief Standard deviation and median absolute value of the nonzero pixels of a stamp (the donut around an island once
 *        its inner box is zeroed), computed with the xtensor expressions the detection loop used.
 * @param stamp `[2 * radius + 1, 2 * radius + 1]` values, row major.
 */
DonutStats stamp_donut_stats(std::span<const float> stamp, long radius);

/**
 * @brief Reduced chi2 of a stamp against the beam kernel, over the nonzero stamp pixels, computed with the xtensor
 *        expressions the detection loop used.
 *
 * The kernel is scaled to the stamp (`sum(stamp) / sum(kernel)`), and the result is `sum(residual^2) / (adjacent_noise^2
 * * count)` over the `count` nonzero pixels. 0 when there is no nonzero pixel or `adjacent_noise` is 0.
 * @param beam_stamp `[2 * halfsize + 1, 2 * halfsize + 1]` values, row major.
 * @param beam_kernel Same shape as `beam_stamp`.
 */
float stamp_beam_chi2(std::span<const float> beam_stamp, std::span<const float> beam_kernel, long halfsize,
                      float adjacent_noise);

} // namespace breifast::plane_lot::detail
