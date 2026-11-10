#pragma once

#include "plane_lot_dispatch.hpp"

namespace breifast::gpu
{
ProcessPlaneResult gpu_process_plane_batch(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                           const PlaneConfig &config);
} // namespace breifast::gpu
