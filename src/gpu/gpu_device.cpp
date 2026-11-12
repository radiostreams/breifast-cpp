#include "gpu_device.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#ifdef __linux__
#include <dlfcn.h>
#endif

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
GpuDeviceSummary gpu_device_summary(int device_id)
{
  cudaDeviceProp prop;
  CHECK_GPU(cudaGetDeviceProperties(&prop, device_id));
  return GpuDeviceSummary{prop.name,
                          prop.major,
                          prop.minor,
                          prop.totalGlobalMem,
                          prop.multiProcessorCount,
                          prop.maxThreadsPerBlock,
                          prop.maxThreadsPerMultiProcessor};
}

namespace
{
// The nix Python interpreter uses nix's glibc, which does not search the host driver
// directories. Load libcuda explicitly so the static CUDA runtime can resolve it.
void preload_cuda_driver()
{
#ifdef __linux__
  if (dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL) != nullptr)
  {
    return;
  }
  constexpr const char *candidates[] = {
      "/usr/lib64/libcuda.so.1",
      "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
      "/run/opengl-driver/lib/libcuda.so.1",
  };
  for (const char *candidate : candidates)
  {
    if (dlopen(candidate, RTLD_NOW | RTLD_GLOBAL) != nullptr)
    {
      return;
    }
  }
#endif
}
} // namespace

void require_cuda_device()
{
  preload_cuda_driver();

  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string("breifast_helpers requires CUDA: ") + cudaGetErrorString(status));
  }
  if (device_count <= 0)
  {
    throw std::runtime_error("breifast_helpers requires CUDA but no device is visible");
  }
}
} // namespace breifast::gpu
