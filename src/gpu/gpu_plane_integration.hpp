#pragma once

#include "plane_lot_dispatch.hpp"

namespace breifast::gpu
{
struct GpuSlot; // gpu_band_result.cuh, which needs the CUDA headers

/// Processes one plane on the GPU with `slot`'s buffers and stream, growing them as needed. The bands run in the CPU
/// order, on the slot's stream.
ProcessPlaneResult gpu_process_plane(GpuSlot &slot, const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                     const PlaneConfig &config);

/// Same, with a slot of its own.
ProcessPlaneResult gpu_process_plane(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                     const PlaneConfig &config);
} // namespace breifast::gpu
