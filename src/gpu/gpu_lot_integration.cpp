#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "gpu_band.hpp"
#include "gpu_plane.hpp"

namespace breifast::gpu
{
namespace
{
// Slots wait here between calls. A plane takes one for its whole run, so a slot never depends on which thread runs it.
struct SlotPool
{
  std::mutex mutex;
  std::vector<std::unique_ptr<GpuSlot>> free;
};

SlotPool &slot_pool()
{
  static SlotPool pool; // built on first use: CUDA is not ready at load time
  return pool;
}

class SlotLease
{
public:
  SlotLease()
  {
    {
      const std::lock_guard lock{slot_pool().mutex};
      if (!slot_pool().free.empty())
      {
        slot_ = std::move(slot_pool().free.back());
        slot_pool().free.pop_back();
      }
    }
    if (!slot_)
    {
      slot_ = std::make_unique<GpuSlot>();
    }
    scope_.emplace(slot_->stream);
  }
  SlotLease(const SlotLease &) = delete;
  SlotLease &operator=(const SlotLease &) = delete;
  ~SlotLease()
  {
    scope_.reset(); // unbind before the slot goes back
    const std::lock_guard lock{slot_pool().mutex};
    slot_pool().free.push_back(std::move(slot_));
  }
  GpuSlot &operator*()
  {
    return *slot_;
  }

private:
  std::unique_ptr<GpuSlot> slot_;
  std::optional<StreamScope> scope_;
};
} // namespace

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

  const GpuSharedInputs shared{groups, config};

  tbb::task_arena arena(slot_count);
  arena.execute(
      [&]
      {
        // A plane that throws cancels the planes not started yet, and parallel_for rethrows its exception once the
        // planes in flight have finished. Each plane in flight holds its own slot.
        tbb::parallel_for(std::size_t{0}, items.size(),
                          [&](const std::size_t index)
                          {
                            SlotLease lease;
                            auto &slot = *lease;
                            results[index] =
                                gpu_process_plane(slot, shared, items[index], config);
                          });
      });
  return results;
}
} // namespace breifast::gpu
