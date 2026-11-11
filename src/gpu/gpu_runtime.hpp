#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace breifast::gpu
{
/// A CUDA runtime call that did not return cudaSuccess.
class GpuError : public std::runtime_error
{
public:
  GpuError(const cudaError_t error, const std::string &message) : std::runtime_error{message}, error_{error} {}

  [[nodiscard]] cudaError_t error() const noexcept { return error_; }

private:
  cudaError_t error_;
};

namespace detail
{
/// True for the errors that CUDA documents as leaving the process unusable (driver_types.h: "the process must be
/// terminated and relaunched"): every later CUDA call fails too.
constexpr bool is_sticky(const cudaError_t error)
{
  switch (error)
  {
  case cudaErrorIllegalAddress:
  case cudaErrorLaunchTimeout:
  case cudaErrorAssert:
  case cudaErrorHardwareStackError:
  case cudaErrorIllegalInstruction:
  case cudaErrorMisalignedAddress:
  case cudaErrorInvalidAddressSpace:
  case cudaErrorInvalidPc:
  case cudaErrorLaunchFailure:
  case cudaErrorMpsClientTerminated:
  case cudaErrorExternalDevice:
    return true;
  default:
    return false;
  }
}

inline void check(const cudaError_t error, const char *const expression, const char *const file, const int line)
{
  if (error != cudaSuccess)
  {
    throw GpuError{error, std::string{file} + ":" + std::to_string(line) + ": " + expression + " failed: " +
                              cudaGetErrorName(error) + ": " + cudaGetErrorString(error) +
                              (is_sticky(error) ? " (the CUDA context is unusable: restart the process)" : "")};
  }
}
} // namespace detail
} // namespace breifast::gpu

/// Throws GpuError when `expression`, a CUDA runtime call, does not return cudaSuccess. Kernel launches are checked
/// with CHECK_GPU(cudaGetLastError()).
#define CHECK_GPU(expression) ::breifast::gpu::detail::check((expression), #expression, __FILE__, __LINE__)

#define gpuMemcpyHostToDeviceAsync(dst, src, size, stream)                                                             \
  CHECK_GPU(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream))
#define gpuMemcpyDeviceToHostAsync(dst, src, size, stream)                                                             \
  CHECK_GPU(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream))
#define gpuMemsetAsync(ptr, value, size, stream) CHECK_GPU(cudaMemsetAsync(ptr, value, size, stream))
#define gpuStreamCreate(stream) CHECK_GPU(cudaStreamCreate(stream))
#define gpuStreamDestroy(stream) CHECK_GPU(cudaStreamDestroy(stream))
#define gpuStreamSynchronize(stream) CHECK_GPU(cudaStreamSynchronize(stream))
