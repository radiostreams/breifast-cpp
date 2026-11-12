#include <algorithm>
#include <cstddef>
#include <vector>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "gpu_band.hpp"
#include "gpu_plane.hpp"

namespace breifast::gpu
{
std::vector<ProcessPlaneResult> run_batch_core_gpu(std::vector<PlaneBatchItem> &items,
                                                   const std::vector<PlaneBatchGroup> &groups,
                                                   const PlaneConfig &config)
{
  const NvtxRange range{"batch"};
  validate_plane_inputs(items, groups, config);
  std::vector<ProcessPlaneResult> results(items.size());
  if (items.empty())
  {
    return results;
  }

  // Populate the per-config cache once before workers read it.
  (void)config.max_to_std_ratio_value();

  // At most slot_count planes are on the GPU at once, each on its own slot. TODO: choose the count from the device
  // memory a slot needs.
  const auto slot_count = static_cast<int>(std::min<std::size_t>(4, items.size()));
  std::vector<GpuSlot> slots(static_cast<std::size_t>(slot_count));
  tbb::task_arena arena(slot_count);
  arena.execute(
      [&]
      {
        // A plane that throws cancels the planes not started yet, and parallel_for rethrows its exception once the
        // planes in flight have finished. Each thread of the arena has its own index in [0, slot_count), so no two
        // planes share a slot.
        tbb::parallel_for(std::size_t{0}, items.size(),
                          [&](const std::size_t index)
                          {
                            auto &slot = slots[static_cast<std::size_t>(tbb::this_task_arena::current_thread_index())];
                            results[index] =
                                gpu_process_plane(slot, items[index], groups[items[index].group_index], config);
                          });
      });
  return results;
}
} // namespace breifast::gpu
