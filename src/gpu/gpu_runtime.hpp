#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <nvtx3/nvToolsExt.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace breifast::gpu
{
/// Whether stream-ordered allocation (cudaMallocAsync) can be used on this device. The first call also tells the
/// device's memory pool to keep the blocks it is given back instead of returning them to the system at the next
/// synchronisation, so that a block freed by one growth is there for the next. Never throws: any failure means false,
/// and callers fall back to cudaMalloc. Assumes one device per process.
inline bool memory_pool_ready()
{
  static const bool ready = []() -> bool
  {
    int device = 0;
    int supported = 0;
    cudaMemPool_t pool = nullptr;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device) != cudaSuccess || supported == 0 ||
        cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess)
    {
      return false;
    }
    std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
    return cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) == cudaSuccess;
  }();
  return ready;
}

/// NVTX domain of the GPU pipeline's ranges, so that Nsight Systems groups and filters them.
inline nvtxDomainHandle_t nvtx_domain()
{
  static const nvtxDomainHandle_t domain = nvtxDomainCreateA("breifast");
  return domain;
}

/// A range on the calling thread's timeline, from construction to destruction. Without a profiler attached, NVTX
/// calls return at once.
class NvtxRange
{
public:
  explicit NvtxRange(const char *name)
  {
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = name;
    nvtxDomainRangePushEx(nvtx_domain(), &attributes);
  }
  explicit NvtxRange(const std::string &name) : NvtxRange(name.c_str())
  {
  }

  NvtxRange(const NvtxRange &) = delete;
  NvtxRange &operator=(const NvtxRange &) = delete;
  ~NvtxRange()
  {
    nvtxDomainRangePop(nvtx_domain());
  }
};

/// A CUDA runtime call that did not return cudaSuccess.
class GpuError : public std::runtime_error
{
public:
  GpuError(const cudaError_t error, const std::string &message) : std::runtime_error{message}, error_{error}
  {
  }

  [[nodiscard]] cudaError_t error() const noexcept
  {
    return error_;
  }

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
    throw GpuError{error, std::string{file} + ":" + std::to_string(line) + ": " + expression +
                              " failed: " + cudaGetErrorName(error) + ": " + cudaGetErrorString(error) +
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
  Stream()
  {
    CHECK_GPU(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }
  Stream(Stream &&other) noexcept : stream_{std::exchange(other.stream_, nullptr)}
  {
  }
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
  ~Stream()
  {
    reset();
  }

  /// Waits for the work enqueued so far; throws GpuError if any of it failed.
  void synchronize() const
  {
    CHECK_GPU(cudaStreamSynchronize(get()));
  }

  [[nodiscard]] cudaStream_t get() const noexcept
  {
    assert(stream_ != nullptr && "moved-from Stream");
    return stream_;
  }
  operator cudaStream_t() const noexcept
  {
    return get();
  }

  static Stream &local_stream()
  {
    if (Stream *bound = bound_stream(); bound != nullptr)
    {
      return *bound;
    }
    thread_local Stream own; // created on first use only
    return own;
  }

private:
  friend class StreamScope;
  static Stream *&bound_stream()
  {
    thread_local Stream *bound = nullptr;
    return bound;
  }

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

/// While alive, local_stream() of the calling thread is `stream`: DeviceVector allocates on it and the plane code runs
/// on it.
class StreamScope
{
public:
  explicit StreamScope(Stream &stream) : previous_{Stream::bound_stream()}
  {
    Stream::bound_stream() = &stream;
  }
  StreamScope(const StreamScope &) = delete;
  StreamScope &operator=(const StreamScope &) = delete;
  ~StreamScope()
  {
    Stream::bound_stream() = previous_;
  }

private:
  Stream *previous_;
};

/// Owning device array, move-only. The implicit pointer conversion keeps kernel call sites short. Memory comes from the
/// device's memory pool (cudaMallocAsync) on the calling thread's local_stream() when the device has one, else from
/// cudaMalloc; it is freed with cudaFree, which also takes pool memory. Copies and memsets run on the stream that
/// allocated the buffer.
template <typename T> class DeviceVector
{
public:
  DeviceVector() = default;

  /// Uninitialised. Assert builds fill it with 0xFF bytes, so that a read before the first write shows up in tests.
  explicit DeviceVector(std::size_t n)
  {
    ensure(n);
    size_ = {n};
  }

  // Written out so that a moved-from vector is empty: defaulted moves would leave its size behind with a null pointer.
  DeviceVector(DeviceVector &&other) noexcept
      : data_{std::exchange(other.data_, nullptr)}, size_{std::exchange(other.size_, 0)},
        allocation_stream_(std::exchange(other.allocation_stream_, nullptr))
  {
  }

  DeviceVector &operator=(DeviceVector &&other) noexcept
  {
    if (this != &other)
    {
      if (data_ != nullptr)
      {
        (void)cudaFree(data_); // synchronous, and valid for memory from cudaMallocAsync too
      }
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      allocation_stream_ = std::exchange(other.allocation_stream_, nullptr);
    }
    return *this;
  }

  ~DeviceVector()
  {
    if (data_ != nullptr)
    {
      (void)cudaFree(data_); // synchronous, and valid for memory from cudaMallocAsync too: the stream may be gone
    }
  }

  static DeviceVector zeros(std::size_t n)
  {
    DeviceVector vector(n);
    if (n > 0)
    {
      // On the stream that allocated the buffer (the default stream for synchronous memory), and waited for there.
      CHECK_GPU(cudaMemsetAsync(vector.data(), 0, n * sizeof(T), vector.allocation_stream_));
      CHECK_GPU(cudaStreamSynchronize(vector.allocation_stream_));
    }
    return vector;
  }

  static DeviceVector from_host(const T *host, std::size_t n)
  {
    DeviceVector vector(n);
    vector.upload(host, n);
    return vector;
  }

  /// Grows to at least n elements; the contents are unspecified after growing. With memory pools, the new buffer is
  /// taken from the pool in the order of the calling thread's stream, and the old one is given back in the order of the
  /// stream that allocated it, with no device-wide synchronisation. That only holds when that stream is the calling
  /// thread's: a buffer grown from another thread, or allocated synchronously, is freed with cudaFree, which waits for
  /// the device. Without pools, everything is synchronous. A buffer from the pool may only be used on its stream.
  void ensure(std::size_t n)
  {
    if (n <= size_)
    {
      return;
    }
    n = std::max(n, 2 * size_);
    const cudaStream_t stream = memory_pool_ready() ? Stream::local_stream().get() : nullptr;
    if (data_ != nullptr)
    {
      if (stream != nullptr && stream == allocation_stream_)
      {
        CHECK_GPU(cudaFreeAsync(data_, allocation_stream_));
      }
      else
      {
        CHECK_GPU(cudaFree(data_));
      }
      data_ = nullptr; // nothing dangles if the allocation below throws
      size_ = 0;
      allocation_stream_ = nullptr;
    }
    void *raw = nullptr;
    if (stream != nullptr)
    {
      CHECK_GPU(cudaMallocAsync(&raw, n * sizeof(T), stream));
    }
    else
    {
      CHECK_GPU(cudaMalloc(&raw, n * sizeof(T)));
    }
    data_ = static_cast<T *>(raw);
    allocation_stream_ = stream;
    size_ = n;
#ifndef NDEBUG
    // Poison, so that a read before the first write shows up in tests. On the allocation's own stream.
    CHECK_GPU(cudaMemsetAsync(data_, 0xFF, n * sizeof(T), allocation_stream_));
    CHECK_GPU(cudaStreamSynchronize(allocation_stream_));
#endif
  }

  /// Grows to at least n elements and copies `host[0, n)` into the first n, on the stream that allocated the buffer:
  /// the work queued on that stream after this call sees the data. `host` may be freed once the call returns: a
  /// pageable source is staged by then.
  void assign(const T *host, std::size_t n)
  {
    ensure(n);
    if (n > 0)
    {
      CHECK_GPU(cudaMemcpyAsync(data(), host, n * sizeof(T), cudaMemcpyHostToDevice, allocation_stream_));
    }
  }

  T *data() noexcept
  {
    return data_;
  }
  const T *data() const noexcept
  {
    return data_;
  }
  operator T *() noexcept
  {
    return data();
  }
  operator const T *() const noexcept
  {
    return data();
  }

  std::size_t size() const noexcept
  {
    return size_;
  }
  bool empty() const noexcept
  {
    return size_ == 0;
  }

private:
  /// Copies and waits for the copy: a pageable host-to-device copy can return before its last DMA lands.
  void upload(const T *host, std::size_t n)
  {
    if (n > 0)
    {
      CHECK_GPU(cudaMemcpyAsync(data(), host, n * sizeof(T), cudaMemcpyHostToDevice, allocation_stream_));
      CHECK_GPU(cudaStreamSynchronize(allocation_stream_)); // the stream the call above ran on
    }
  }

  T *data_ = nullptr;
  std::size_t size_ = 0;
  cudaStream_t allocation_stream_ = nullptr;
};
} // namespace breifast::gpu
