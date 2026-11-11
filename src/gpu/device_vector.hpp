#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>

#include <cuda_runtime.h>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
/// Owning device array: plain cudaMalloc/cudaFree, move-only. The implicit pointer conversion keeps kernel call sites
/// short. Uploads go through the legacy default stream and wait for it, so their data is in place before any other
/// stream, blocking or not, can read it.
template <typename T> class DeviceVector
{
public:
  DeviceVector() = default;

  /// Uninitialised. Assert builds fill it with 0xFF bytes, so that a read before the first write shows up in tests.
  explicit DeviceVector(std::size_t n) : data_{allocate(n)}, size_{n} {}

  static DeviceVector zeros(std::size_t n)
  {
    DeviceVector vector(n);
    if (n > 0)
    {
      CHECK_GPU(cudaMemset(vector.data(), 0, n * sizeof(T)));
      CHECK_GPU(cudaStreamSynchronize(cudaStreamLegacy));
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

  T *data() noexcept { return data_.get(); }
  const T *data() const noexcept { return data_.get(); }
  operator T *() noexcept { return data(); }
  operator const T *() const noexcept { return data(); }

  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  // Transitional, for the tests until step 9 replaces them with upload helpers: host staging, then an upload.
  DeviceVector(std::size_t n, const T &value) : DeviceVector(n) { assign(n, value); }

  template <std::input_iterator InputIt> DeviceVector(InputIt first, InputIt last)
      : DeviceVector(static_cast<std::size_t>(std::distance(first, last)))
  {
    const auto staging = std::make_unique<T[]>(size_);
    std::copy(first, last, staging.get());
    upload(staging.get(), size_);
  }

  void assign(std::size_t n, const T &value)
  {
    ensure(n);
    const auto staging = std::make_unique<T[]>(n);
    std::fill_n(staging.get(), n, value);
    upload(staging.get(), n);
  }

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
    CHECK_GPU(cudaStreamSynchronize(cudaStreamLegacy));
#endif
    return pointer;
  }

  /// A pageable host-to-device cudaMemcpy can return before its last DMA lands, so wait for the legacy stream.
  void upload(const T *host, std::size_t n)
  {
    if (n > 0)
    {
      CHECK_GPU(cudaMemcpy(data(), host, n * sizeof(T), cudaMemcpyHostToDevice));
      CHECK_GPU(cudaStreamSynchronize(cudaStreamLegacy));
    }
  }

  Pointer data_;
  std::size_t size_ = 0;
};
} // namespace breifast::gpu
