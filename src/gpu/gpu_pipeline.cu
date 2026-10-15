#include "gpu_pipeline.hpp"

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
GpuDeviceSummary gpu_device_summary(int device_id)
{
  cudaDeviceProp prop;
  CHECK_GPU_ERROR(cudaGetDeviceProperties(&prop, device_id));
  return GpuDeviceSummary{prop.name,
                          prop.major,
                          prop.minor,
                          prop.totalGlobalMem,
                          prop.multiProcessorCount,
                          prop.maxThreadsPerBlock,
                          prop.maxThreadsPerMultiProcessor};
}
} // namespace breifast::gpu
