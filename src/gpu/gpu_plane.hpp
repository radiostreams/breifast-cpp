#pragma once

#include <vector>

#include "plane_lot_dispatch.hpp"
#include "process_plane_result.hpp"

// Public and CUDA-free: the extension module includes it.
namespace breifast::gpu
{
struct GpuSlot; // gpu_band.hpp, which needs the CUDA headers
struct GpuSharedInputs;

/// Processes one plane on the GPU with `slot`'s buffers and stream, growing them as needed. The bands run in the CPU
/// order, on the slot's stream.
ProcessPlaneResult gpu_process_plane(GpuSlot &slot, const GpuSharedInputs &shared, const PlaneBatchItem &item,
                                     const PlaneConfig &config);

/// Same, with a slot of its own.
ProcessPlaneResult gpu_process_plane(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                     const PlaneConfig &config);

/// Runs every plane of `items` on the GPU, after validate_plane_inputs(). If a plane throws, the planes not started yet
/// are skipped and its exception is rethrown once the planes in flight have finished.
std::vector<ProcessPlaneResult> run_batch_core_gpu(std::vector<PlaneBatchItem> &items,
                                                   const std::vector<PlaneBatchGroup> &groups,
                                                   const PlaneConfig &config);
} // namespace breifast::gpu
