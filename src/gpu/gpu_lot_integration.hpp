#pragma once

#include <vector>

#include "plane_lot_dispatch.hpp"
#include "process_plane_result.hpp"

namespace breifast::gpu
{
std::vector<ProcessPlaneResult> run_batch_core_gpu(std::vector<PlaneBatchItem> &items,
                                                   const std::vector<PlaneBatchGroup> &groups,
                                                   const PlaneConfig &config);
} // namespace breifast::gpu
