#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include <cuda_runtime.h>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
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
