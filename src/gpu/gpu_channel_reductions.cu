#include <stdexcept>

#include "gpu_pipeline.hpp"
#include "gpu_runtime.hpp"

namespace breifast::gpu
{
namespace
{
// TODO: out_mask[p] = OR over c in [band_start, band_stop) of accept_mask[c * n_pixels + p], and *out_any set when
// any out_mask[p] is true.
__global__ void mask_or_kernel(const bool *accept_mask, size_t n_pixels, int band_start, int band_stop, bool *out_mask,
                               int *out_any)
{
}
} // namespace

void gpu_mask_or(const bool *accept_mask, size_t n_pixels, int band_start, int band_stop, bool *out_mask, int *out_any,
                 GpuStream stream)
{
  // TODO: zero *out_any on the stream, pick a launch configuration, launch mask_or_kernel, check the launch.
  (void)accept_mask, (void)n_pixels, (void)band_start, (void)band_stop, (void)out_mask, (void)out_any, (void)stream;
  throw std::logic_error("gpu_mask_or: not implemented");
}
} // namespace breifast::gpu
