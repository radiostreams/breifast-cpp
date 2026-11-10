#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breifast::gpu::detail
{
inline void check(cudaError_t err, char const *func, char const *file, int line)
{
  if (err != cudaSuccess)
  {
    std::stringstream ss;
    ss << "CUDA Runtime Error at: " << file << ":" << line << '\n';
    ss << cudaGetErrorString(err) << " " << func << '\n';
    throw std::runtime_error(ss.str());
  }
}

inline void checkLast(char const *file, int line)
{
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess)
  {
    std::stringstream ss;
    ss << "CUDA Runtime Error at: " << file << ":" << line << '\n';
    ss << cudaGetErrorString(err) << '\n';
    throw std::runtime_error(ss.str());
  }
}
} // namespace breifast::gpu::detail

#define CHECK_GPU_ERROR(val) ::breifast::gpu::detail::check((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_GPU_ERROR() ::breifast::gpu::detail::checkLast(__FILE__, __LINE__)

#define gpuMemcpyHostToDeviceAsync(dst, src, size, stream)                                                             \
  CHECK_GPU_ERROR(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream))
#define gpuMemcpyDeviceToHostAsync(dst, src, size, stream)                                                             \
  CHECK_GPU_ERROR(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream))
#define gpuMemsetAsync(ptr, value, size, stream) CHECK_GPU_ERROR(cudaMemsetAsync(ptr, value, size, stream))
#define gpuStreamCreate(stream) CHECK_GPU_ERROR(cudaStreamCreate(stream))
#define gpuStreamDestroy(stream) CHECK_GPU_ERROR(cudaStreamDestroy(stream))
#define gpuStreamSynchronize(stream) CHECK_GPU_ERROR(cudaStreamSynchronize(stream))
