/**
 * @file islands.hpp
 * @brief Connected-component (island) labeling for 2D boolean detection masks.
 *
 * Public interface for turning a boolean mask into a labeled image, where every 4-connected group of `true` pixels gets
 * a unique positive integer ID and `false` pixels are labeled `0`.
 */
#pragma once
#include <cstddef>
#include <limits>
#include <span>
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
  std::vector<PeakInfo> peak_per_label;

private:
  std::vector<size_t> parents;
  std::vector<size_t> root_to_compact_id;

  friend void label_islands(IslandLabels &out, const xt::xtensor<bool, 2> &masks, const xt::xtensor<float, 2> &image,
                            const xt::xtensor<float, 2> &local_noise);
};

/**
 * @brief Label every 4-connected group of `true` pixels in `image` with a unique ID.
 *
 * @param out IslandLabels with `labels` sized like `image` and `count` set to the number of distinct islands found by
 * this function.
 * @param masks 2D boolean mask (row index = axis 0 / Y, column index = axis 1 / X).
 * @param image 2D double image (row index = axis 0 / Y, column index = axis 1 / X).
 * @param local_noise 2D local noise image (row index = axis 0 / Y, column index = axis 1 / X).
 */
void label_islands(IslandLabels &out, const xt::xtensor<bool, 2> &masks, const xt::xtensor<float, 2> &image,
                   const xt::xtensor<float, 2> &local_noise);

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

/**
 * @brief Flux boost of one island: the ratio `flux / baseline_flux` with the largest absolute value among the island's
 *        pixels, computed with the xtensor expressions the detection loop used.
 *
 * The first pixel in row-major order wins a tie. Scans the whole label image.
 * @param labels `[n_y, n_x]` island labels from label_islands(), row major.
 * @param baseline_flux `[n_y, n_x]` baseline flux, row major.
 * @param label Island to evaluate, `1..count`. A label without any pixel throws.
 * @param flux Flux of the island, the numerator of every ratio.
 */
float island_boost(std::span<const size_t> labels, std::span<const float> baseline_flux, size_t n_y, size_t n_x,
                   size_t label, float flux);
} // namespace breifast::islands
