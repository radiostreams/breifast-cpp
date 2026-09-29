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
#include <tracy/Tracy.hpp>

#include "islands.hpp"

namespace breifast::islands
{
void label_islands(IslandLabels &out, const xt::xtensor<bool, 2> &masks, const xt::xtensor<float, 2> &image,
                   const xt::xtensor<float, 2> &local_noise)
{
  ZoneScoped;
  auto &labels = out.labels;
  auto &parents = out.parents;
  labels.resize(masks.shape());

  const auto n_y = masks.shape(0);
  const auto n_x = masks.shape(1);
  parents.clear();
  parents.push_back(0);

  const auto find_root = [&](const size_t index)
  {
    assert(index < parents.size());
    auto current_index = index;
    auto current_parent = parents[index];
    while (current_index != current_parent)
    {
      current_index = current_parent;
      current_parent = parents[current_index];
    }

    const auto root = current_parent;

    current_index = index;
    current_parent = parents[index];
    while (current_index != current_parent)
    {
      const size_t prev_index = current_index;
      current_index = current_parent;
      current_parent = parents[current_index];
      parents[prev_index] = root;
    }

    return root;
  };

  const auto unite = [&](const size_t index0, const size_t index1)
  {
    const size_t root0 = find_root(index0);
    const size_t root1 = find_root(index1);
    if (root0 != root1)
    {
      parents[root1] = root0;
    }
  };

  // Pass 1: assign provisional labels by looking at each pixel's up/left neighbors (the only two already visited in
  // raster-scan order for 4-connectivity), merging equivalent labels via unite() as needed.
  for (size_t i_y = 0; i_y < n_y; ++i_y)
  {
    for (size_t i_x = 0; i_x < n_x; ++i_x)
    {
      const auto mask = masks(i_y, i_x);
      if (!mask)
      {
        labels(i_y, i_x) = 0;
        continue;
      }

      const auto up = (i_y > 0) ? labels(i_y - 1, i_x) : 0;
      const auto left = (i_x > 0) ? labels(i_y, i_x - 1) : 0;

      if (up == 0 && left == 0)
      {
        labels(i_y, i_x) = parents.size();
        parents.push_back(parents.size());
        continue;
      }

      if (up == 0 || left == 0 || left == up)
      {
        labels(i_y, i_x) = std::max(up, left);
        continue;
      }

      labels(i_y, i_x) = up;
      unite(up, left);
    }
  }

  // Pass 2: resolve every pixel's provisional label to its union-find root, and compact the root values into a
  // 1..count range.
  auto &root_to_compact_id = out.root_to_compact_id;
  root_to_compact_id.assign(parents.size(), 0);
  auto &peaks = out.peak_per_label;
  peaks.clear();
  for (size_t i_y = 0; i_y < n_y; ++i_y)
  {
    for (size_t i_x = 0; i_x < n_x; ++i_x)
    {
      auto &label = labels(i_y, i_x);
      if (!label)
      {
        continue;
      }
      const auto root = find_root(label);
      if (root_to_compact_id[root] == 0)
      {
        peaks.emplace_back();
        root_to_compact_id[root] = peaks.size();
      }
      label = root_to_compact_id[root];

      auto &peak = peaks[label - 1];
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
}

IslandLabels label_islands(const xt::xtensor<bool, 2> &masks, const xt::xtensor<float, 2> &image,
                           const xt::xtensor<float, 2> &local_noise)
{
  IslandLabels result;
  label_islands(result, masks, image, local_noise);
  return result;
}
} // namespace breifast::islands