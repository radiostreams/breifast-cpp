#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace breifast::gpu
{
// Owning device buffer: construction allocates, destruction frees. The implicit
// pointer conversion keeps kernel call sites unchanged. bool is stored as bytes
// because thrust::device_vector<bool> is bit-packed.
template <typename T> class DeviceVector
{
public:
  using Storage = std::conditional_t<std::is_same_v<T, bool>, unsigned char, T>;

  DeviceVector() = default;
  explicit DeviceVector(std::size_t n) : data_(n) {}
  DeviceVector(std::size_t n, const T &value) : data_(n, static_cast<Storage>(value)) {}
  DeviceVector(const T *host, std::size_t n) : data_(n)
  {
    if (n > 0 && cudaMemcpy(data(), host, n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
    {
      throw std::runtime_error("DeviceVector: host-to-device copy failed");
    }
  }
  template <typename InputIt> DeviceVector(InputIt first, InputIt last) : data_(first, last) {}

  DeviceVector(const DeviceVector &) = delete;
  DeviceVector &operator=(const DeviceVector &) = delete;
  DeviceVector(DeviceVector &&) noexcept = default;
  DeviceVector &operator=(DeviceVector &&) noexcept = default;
  ~DeviceVector() = default;

  T *data()
  {
    return reinterpret_cast<T *>(thrust::raw_pointer_cast(data_.data()));
  }
  const T *data() const
  {
    return reinterpret_cast<const T *>(thrust::raw_pointer_cast(data_.data()));
  }
  operator T *()
  {
    return data();
  }
  operator const T *() const
  {
    return data();
  }

  auto begin()
  {
    return data_.begin();
  }
  auto end()
  {
    return data_.end();
  }
  auto begin() const
  {
    return data_.begin();
  }
  auto end() const
  {
    return data_.end();
  }

  std::size_t size() const
  {
    return data_.size();
  }
  bool empty() const
  {
    return data_.empty();
  }
  void resize(std::size_t n)
  {
    data_.resize(n);
  }
  void assign(std::size_t n, const T &value)
  {
    data_.assign(n, static_cast<Storage>(value));
  }

private:
  thrust::device_vector<Storage> data_;
};
} // namespace breifast::gpu
