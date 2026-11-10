#include "gpu_lot_integration.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <oneapi/tbb/parallel_for.h>

#include "gpu_plane_integration.hpp"

namespace breifast::gpu
{
std::vector<ProcessPlaneResult> run_batch_core_gpu(std::vector<PlaneBatchItem> &items,
                                                   const std::vector<PlaneBatchGroup> &groups,
                                                   const PlaneConfig &config)
{
  std::vector<ProcessPlaneResult> results(items.size());
  if (items.empty())
  {
    return results;
  }

  // Populate the per-config cache once before workers read it.
  (void)config.max_to_std_ratio_value();

  const auto process_one = [&](const std::size_t index)
  {
    auto &item = items[index];
    if (item.group_index >= groups.size())
    {
      std::cerr << "group_index " << item.group_index << " exceeds group count " << groups.size() << std::endl;
      return;
    }

    try
    {
      results[index] = gpu_process_plane_batch(item, groups[item.group_index], config);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << std::endl;
    }
  };

  tbb::parallel_for(std::size_t{0}, items.size(), [&](const std::size_t index) { process_one(index); });
  return results;
}
} // namespace breifast::gpu
