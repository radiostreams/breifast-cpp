#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <memory>
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

/// Owning device array: plain cudaMalloc/cudaFree, move-only. The implicit pointer conversion keeps kernel call sites
/// short. Its copies and memsets run on the default stream, which the non-blocking work streams do not wait for, and
/// neither is complete when it returns: each is followed by a sync of the default stream, so the data is in place
/// before any stream can read it.
template <typename T> class DeviceVector
{
public:
  DeviceVector() = default;

  /// Uninitialised. Assert builds fill it with 0xFF bytes, so that a read before the first write shows up in tests.
  explicit DeviceVector(std::size_t n) : data_{allocate(n)}, size_{n} {}

  // Written out so that a moved-from vector is empty: defaulted moves would leave its size behind with a null pointer.
  DeviceVector(DeviceVector &&other) noexcept : data_{std::move(other.data_)}, size_{std::exchange(other.size_, 0)} {}
  DeviceVector &operator=(DeviceVector &&other) noexcept
  {
    if (this != &other)
    {
      data_ = std::move(other.data_);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  static DeviceVector zeros(std::size_t n)
  {
    DeviceVector vector(n);
    if (n > 0)
    {
      CHECK_GPU(cudaMemset(vector.data(), 0, n * sizeof(T)));
      CHECK_GPU(cudaStreamSynchronize(nullptr)); // the default stream, which the call above ran on
    }
    return vector;
  }

  static DeviceVector from_host(const T *host, std::size_t n)
  {
    DeviceVector vector(n);
    vector.upload(host, n);
    return vector;
  }

  /// Grows to at least n elements; the contents are unspecified after growing.
  void ensure(std::size_t n)
  {
    if (n > size_)
    {
      data_.reset();
      size_ = 0;
      data_ = allocate(n);
      size_ = n;
    }
  }

  /// Grows to at least n elements and copies `host[0, n)` into the first n.
  void assign(const T *host, std::size_t n)
  {
    ensure(n);
    upload(host, n);
  }

  T *data() noexcept { return data_.get(); }
  const T *data() const noexcept { return data_.get(); }
  operator T *() noexcept { return data(); }
  operator const T *() const noexcept { return data(); }

  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

private:
  struct Free
  {
    void operator()(T *pointer) const noexcept { (void)cudaFree(pointer); }
  };
  using Pointer = std::unique_ptr<T[], Free>;

  static Pointer allocate(std::size_t n)
  {
    if (n == 0)
    {
      return nullptr;
    }
    void *raw = nullptr;
    CHECK_GPU(cudaMalloc(&raw, n * sizeof(T)));
    Pointer pointer{static_cast<T *>(raw)};
#ifndef NDEBUG
    CHECK_GPU(cudaMemset(pointer.get(), 0xFF, n * sizeof(T)));
    CHECK_GPU(cudaStreamSynchronize(nullptr)); // the default stream, which the call above ran on
#endif
    return pointer;
  }

  /// A pageable host-to-device cudaMemcpy can return before its last DMA lands.
  void upload(const T *host, std::size_t n)
  {
    if (n > 0)
    {
      CHECK_GPU(cudaMemcpy(data(), host, n * sizeof(T), cudaMemcpyHostToDevice));
      CHECK_GPU(cudaStreamSynchronize(nullptr)); // the default stream, which the call above ran on
    }
  }

  Pointer data_;
  std::size_t size_ = 0;
};
} // namespace breifast::gpu
