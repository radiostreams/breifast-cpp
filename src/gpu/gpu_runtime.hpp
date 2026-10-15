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

#define gpuMalloc(ptr, size) CHECK_GPU_ERROR(cudaMalloc(ptr, size))
#define gpuFree(ptr) CHECK_GPU_ERROR(cudaFree(ptr))
#define gpuMallocHost(ptr, size) CHECK_GPU_ERROR(cudaMallocHost(ptr, size))
#define gpuFreeHost(ptr) CHECK_GPU_ERROR(cudaFreeHost(ptr))
#define gpuMemcpy(dst, src, size, kind) CHECK_GPU_ERROR(cudaMemcpy(dst, src, size, kind))
#define gpuMemcpyAsync(dst, src, size, kind, stream) CHECK_GPU_ERROR(cudaMemcpyAsync(dst, src, size, kind, stream))
#define gpuMemset(ptr, value, size) CHECK_GPU_ERROR(cudaMemset(ptr, value, size))
#define gpuMemcpyToSymbol(symbol, src, size, offset, kind)                                                             \
  CHECK_GPU_ERROR(cudaMemcpyToSymbol(symbol, src, size, offset, kind))
#define gpuMemcpyToSymbolAsync(symbol, src, size, offset, kind, stream)                                                \
  CHECK_GPU_ERROR(cudaMemcpyToSymbolAsync(symbol, src, size, offset, kind, stream))
#define gpuMemcpyFromSymbol(dst, symbol, size, offset, kind)                                                           \
  CHECK_GPU_ERROR(cudaMemcpyFromSymbol(dst, symbol, size, offset, kind))
#define gpuMemcpyFromSymbolAsync(dst, symbol, size, offset, kind, stream)                                              \
  CHECK_GPU_ERROR(cudaMemcpyFromSymbolAsync(dst, symbol, size, offset, kind, stream))
#define gpuStreamCreate(stream) CHECK_GPU_ERROR(cudaStreamCreate(stream))
#define gpuStreamDestroy(stream) CHECK_GPU_ERROR(cudaStreamDestroy(stream))
#define gpuStreamSynchronize(stream) CHECK_GPU_ERROR(cudaStreamSynchronize(stream))
#define gpuEventCreate(event) CHECK_GPU_ERROR(cudaEventCreate(event))
#define gpuEventDestroy(event) CHECK_GPU_ERROR(cudaEventDestroy(event))
#define gpuEventRecord(event, stream) CHECK_GPU_ERROR(cudaEventRecord(event, stream))
#define gpuEventSynchronize(event) CHECK_GPU_ERROR(cudaEventSynchronize(event))
#define gpuMemcpyHostToDevice(dst, src, size) gpuMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define gpuMemcpyDeviceToHost(dst, src, size) gpuMemcpy(dst, src, size, cudaMemcpyDeviceToHost)
#define gpuMemcpyDeviceToDevice(dst, src, size) gpuMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)
#define gpuMemcpyHostToDeviceAsync(dst, src, size, stream)                                                             \
  gpuMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream)
#define gpuMemcpyDeviceToHostAsync(dst, src, size, stream)                                                             \
  gpuMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream)
#define gpuMemcpyDeviceToDeviceAsync(dst, src, size, stream)                                                           \
  gpuMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream)
