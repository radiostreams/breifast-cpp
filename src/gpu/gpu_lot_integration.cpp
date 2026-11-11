#include "gpu_lot_integration.hpp"

#include <cstddef>
#include <vector>

#include <oneapi/tbb/parallel_for.h>

#include "gpu_plane_integration.hpp"

namespace breifast::gpu
{
std::vector<ProcessPlaneResult> run_batch_core_gpu(std::vector<PlaneBatchItem> &items,
                                                   const std::vector<PlaneBatchGroup> &groups,
                                                   const PlaneConfig &config)
{
  validate_plane_inputs(items, groups, config);
  std::vector<ProcessPlaneResult> results(items.size());
  if (items.empty())
  {
    return results;
  }

  // Populate the per-config cache once before workers read it.
  (void)config.max_to_std_ratio_value();

  // A plane that throws cancels the planes not started yet, and parallel_for rethrows its exception once the planes in
  // flight have finished.
  tbb::parallel_for(
      std::size_t{0}, items.size(), [&](const std::size_t index)
      { results[index] = gpu_process_plane_batch(items[index], groups[items[index].group_index], config); });
  return results;
}
} // namespace breifast::gpu
