#pragma once

#include "gpu_band_workspace.cuh"
#include "gpu_runtime.hpp"

namespace breifast::gpu
{
/// Runs one band on `stream` and leaves its outputs in `result`. Returns false when the band's accept mask is empty:
/// the band is skipped, as on the CPU. The caller has already skipped a band with no nonzero channel.
bool gpu_process_band(const GpuBandInput &input, GpuBandWorkspace &workspace, GpuBandResult &result, GpuStream stream);
} // namespace breifast::gpu
