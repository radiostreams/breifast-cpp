/**
 * @file islands.hpp
 * @brief Connected-component (island) labeling for 2D boolean detection masks.
 *
 * Public interface for turning a boolean mask into a labeled image, where every 4-connected group of `true` pixels gets
 * a unique positive integer ID and `false` pixels are labeled `0`.
 */
#pragma once
#include <limits>
#include <unordered_map>
#include <xtensor/containers/xtensor.hpp>

namespace breifast::islands
{
/**
 * @brief Result of label_islands(): a labeled image plus how many islands it contains.
 */
struct IslandLabels
{
  /// Same shape as the input mask. `0` = background, `1..count` = island IDs.
  xt::xtensor<size_t, 2> labels;

  struct PeakInfo
  {
    float value = std::numeric_limits<float>::lowest();
    float local_noise = std::numeric_limits<float>::lowest();
    std::pair<size_t, size_t> coordinates;
    size_t island_size = 0;
  };
  std::unordered_map<size_t, PeakInfo> peak_per_label;
};

/**
 * @brief Label every 4-connected group of `true` pixels in `image` with a unique ID.
 *
 * @param masks 2D boolean mask (row index = axis 0 / Y, column index = axis 1 / X).
 * @param image 2D double image (row index = axis 0 / Y, column index = axis 1 / X).
 * @param local_noise 2D local noise image (row index = axis 0 / Y, column index = axis 1 / X).
 * @return An IslandLabels with `labels` sized like `image` and `count` set to the number of distinct islands found.
 */
IslandLabels label_islands(const xt::xtensor<bool, 2> &masks, const xt::xtensor<float, 2> &image,
                           const xt::xtensor<float, 2> &local_noise);
} // namespace breifast::islands