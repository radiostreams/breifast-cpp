#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include <oneapi/tbb/task_group.h>

#include "gpu_band.hpp"
#include "gpu_plane.hpp"

namespace breifast::gpu
{
std::vector<ProcessPlaneResult> run_batch_core_gpu(std::vector<PlaneBatchItem> &items,
                                                   const std::vector<PlaneBatchGroup> &groups,
                                                   const PlaneConfig &config,
                                                   const std::function<void(std::size_t)> &prepare)
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

  // The device buffers are kept from call to call, so that a call allocates only what the earlier ones did not. One
  // call at a time: concurrent calls would only compete for the GPU, and each would need buffers of its own.
  static std::mutex mutex;
  static GpuSharedInputs shared;
  static std::vector<GpuSlot> slots;
  const std::lock_guard lock{mutex};

  shared.assign(groups, config);

  // At most slot_count planes are on the GPU at once. TODO: choose the count from the device memory a slot needs.
  // BREIFAST_GPU_SLOTS overrides the 16, for experiments.
  const char *const slots_env = std::getenv("BREIFAST_GPU_SLOTS");
  const std::size_t max_slots = slots_env != nullptr ? std::stoul(slots_env) : 16;
  const std::size_t slot_count = std::min<std::size_t>(max_slots, items.size());
  if (slots.size() < slot_count)
  {
    slots.resize(slot_count);
  }

  // One task per slot, each taking the next plane until none is left, so that a slot serves one plane at a time. A
  // plane that throws stops the planes not started yet, and wait() rethrows its exception once the others have
  // finished.
  std::atomic<std::size_t> next{0};
  tbb::task_group tasks;
  for (auto &slot : std::span{slots}.first(slot_count))
  {
    tasks.run(
        [&]
        {
          const StreamScope scope{slot.stream};
          for (std::size_t index = next++; index < items.size() && !tbb::is_current_task_group_canceling();
               index = next++)
          {
            if (prepare)
            {
              prepare(index);
            }
            results[index] = gpu_process_plane(slot, shared, items[index], config);
          }
        });
  }
  tasks.wait();
  return results;
}
} // namespace breifast::gpu
