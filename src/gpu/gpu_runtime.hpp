#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>

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

namespace breifast::gpu
{
/// Owns a non-blocking CUDA stream. The destructor waits for the stream's work before destroying it, so a Stream
/// declared after the buffers its work uses is destroyed first and nothing is freed while still in use.
class Stream
{
public:
  Stream() { CHECK_GPU(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  Stream(Stream &&other) noexcept : stream_{std::exchange(other.stream_, nullptr)} {}
  Stream &operator=(Stream &&other) noexcept
  {
    if (this != &other)
    {
      reset();
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  ~Stream() { reset(); }

  /// Waits for the work enqueued so far; throws GpuError if any of it failed.
  void synchronize() const { CHECK_GPU(cudaStreamSynchronize(get())); }

  [[nodiscard]] cudaStream_t get() const noexcept
  {
    assert(stream_ != nullptr && "moved-from Stream");
    return stream_;
  }
  operator cudaStream_t() const noexcept { return get(); }

private:
  /// Never throws: an error here comes from earlier work, which synchronize() reports.
  void reset() noexcept
  {
    if (stream_ != nullptr)
    {
      (void)cudaStreamSynchronize(stream_);
      (void)cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  cudaStream_t stream_ = nullptr;
};
} // namespace breifast::gpu
