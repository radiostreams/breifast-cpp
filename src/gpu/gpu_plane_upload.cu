#include <cuda_runtime.h>

#include <stdexcept>

#include "gpu_plane_jobs.hpp"

namespace breifast::gpu
{
void *gpu_upload_buffer(const void *host, std::size_t bytes)
{
  if (bytes == 0)
  {
    return nullptr;
  }
  if (host == nullptr)
  {
    throw std::runtime_error("gpu_upload_buffer: null host pointer");
  }

  void *device = nullptr;
  if (cudaMalloc(&device, bytes) != cudaSuccess)
  {
    throw std::runtime_error("gpu_upload_buffer: cudaMalloc failed");
  }
  if (cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    cudaFree(device);
    throw std::runtime_error("gpu_upload_buffer: cudaMemcpy failed");
  }
  return device;
}

void gpu_free_buffer(void *device) noexcept
{
  if (device != nullptr)
  {
    cudaFree(device);
  }
}
} // namespace breifast::gpu
