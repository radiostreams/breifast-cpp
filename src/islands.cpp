/**
 * @file islands.cpp
 * @brief Two-pass union-find implementation of label_islands().
 *
 * - Pass 1 walks the mask in raster order, assigning each `true` pixel a provisional label by looking at its
 *   already-visited up/left neighbors, and recording equivalences (via union-find) whenever two differently-labeled
 *   neighbors turn out to be the same island.
 * - Pass 2 resolves every pixel's provisional label to its canonical union-find root and compacts the (possibly sparse)
 *   root values into a dense `1..count` range.
 */
#include <cassert>
#include <numeric>

#include "islands.hpp"

namespace breifast::islands
{
/**
 * @brief Union-find-backed worker for one label_islands() call.
 */
struct IslandsLabeler
{
public:
  IslandsLabeler() = default;
  /**
   * @brief Run both passes of the labeling algorithm on `image`.
   * @param image 2D boolean mask, axis 0 = Y (rows), axis 1 = X (columns).
   * @return Labeled result; see IslandLabels.
   */
  IslandLabels operator()(const xt::xtensor<bool, 2> &masks, const xt::xtensor<float, 2> &image,
                          const xt::xtensor<float, 2> &local_noise);

private:
  /**
   * @brief Union-find `find`, with path compression.
   *
   * Walks parents_ up from `index` until it reaches a label that is its own parent (the root), then re-walks the same
   * path so every visited label points directly at that root, making the next find() on any of them O(1).
   * @param index A provisional (or already-resolved) label ID.
   * @return The canonical root label for `index`'s island.
   */
  size_t find_root(const size_t index)
  {
    assert(index < parents_.size());
    auto current_index = index;
    auto current_parent = parents_[index];
    while (current_index != current_parent)
    {
      current_index = current_parent;
      current_parent = parents_[current_index];
    }

    const auto root = current_parent;

    current_index = index;
    current_parent = parents_[index];
    while (current_index != current_parent)
    {
      const size_t prev_index = current_index;
      current_index = current_parent;
      current_parent = parents_[current_index];
      parents_[prev_index] = root;
    }

    return root;
  }

  /**
   * @brief Union-find `unite`: mark index0's and index1's islands as the same one.
   *
   * Unifies pixels by setting one as a parent to the other.
   * @param index0 A provisional label.
   * @param index1 Another provisional label, to be merged with index0's island.
   */
  void unite(const size_t index0, const size_t index1)
  {
    const size_t root0 = find_root(index0);
    const size_t root1 = find_root(index1);
    if (root0 != root1)
    {
      parents_[root1] = root0;
    }
  };

  /**
   * @brief Clears the temporary data structures. Memory is not overwritten.
   */
  void clear()
  {
    parents_.clear();
    labels_.resize({0, 0});
  }

  /// Union-find parent pointers, indexed by provisional label ID. Index `0` is an unused placeholder so real label IDs.
  /// (starting at `1`) map directly onto it.
  std::vector<size_t> parents_;
  ///  Working label grid for the current image; moved into the returned IslandLabels::labels at the end of operator().
  xt::xtensor<size_t, 2> labels_;
  std::unordered_map<size_t, IslandLabels::PeakInfo> peak_per_label_;
};

IslandLabels IslandsLabeler::operator()(const xt::xtensor<bool, 2> &masks, const xt::xtensor<float, 2> &image,
                                        const xt::xtensor<float, 2> &local_noise)
{
  labels_.resize(masks.shape());
  labels_.fill(0);

  size_t next_label = 1;

  const auto n_y = masks.shape(0);
  const auto n_x = masks.shape(1);
  parents_.resize(n_y * n_x + 1);
  std::iota(parents_.begin(), parents_.end(), 0);

  // Pass 1: assign provisional labels by looking at each pixel's up/left neighbors (the only two already visited in
  // raster-scan order for 4-connectivity), merging equivalent labels via unite() as needed.
  for (size_t i_y = 0; i_y < n_y; ++i_y)
  {
    for (size_t i_x = 0; i_x < n_x; ++i_x)
    {
      const auto mask = masks(i_y, i_x);
      if (!mask)
      {
        continue;
      }

      const auto up = (i_y > 0) ? labels_(i_y - 1, i_x) : 0;
      const auto left = (i_x > 0) ? labels_(i_y, i_x - 1) : 0;

      if (up == 0 && left == 0)
      {
        labels_(i_y, i_x) = next_label++;
        continue;
      }

      if (up == 0 || left == 0 || left == up)
      {
        labels_(i_y, i_x) = std::max(up, left);
        continue;
      }

      labels_(i_y, i_x) = up;
      unite(up, left);
    }
  }

  // Pass 2: resolve every pixel's provisional label to its union-find root, and compact the root values into a
  // 1..count range.
  std::vector<size_t> root_to_compact_id(parents_.size(), 0);
  size_t compact_count = 0;
  for (size_t i_y = 0; i_y < n_y; ++i_y)
  {
    for (size_t i_x = 0; i_x < n_x; ++i_x)
    {
      auto &label = labels_(i_y, i_x);
      if (!label)
      {
        continue;
      }
      const auto root = find_root(label);
      if (root_to_compact_id[root] == 0)
      {
        root_to_compact_id[root] = ++compact_count;
      }
      const auto compact_label = root_to_compact_id[root];
      labels_(i_y, i_x) = compact_label;

      auto &peak = peak_per_label_[compact_label];
      peak.island_size++;
      const auto abs_value = std::abs(image(i_y, i_x));
      peak.local_noise = std::max(peak.local_noise, local_noise(i_y, i_x));
      if (abs_value > peak.value)
      {
        peak.value = abs_value;
        peak.coordinates.first = i_y;
        peak.coordinates.second = i_x;
      }
    }
  }

  IslandLabels result;
  result.labels = std::move(labels_);
  result.peak_per_label = std::move(peak_per_label_);
  return result;
}

IslandLabels label_islands(const xt::xtensor<bool, 2> &masks, const xt::xtensor<float, 2> &image,
                           const xt::xtensor<float, 2> &local_noise)
{
  return IslandsLabeler{}(masks, image, local_noise);
}
} // namespace breifast::islands