#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "device_vector.hpp"
#include "gpu_runtime.hpp"

/// Host-device copies for the GPU tests. Uploads are in place before any stream can read them (DeviceVector::from_host);
/// downloads wait for the work enqueued on their stream so far.
namespace breifast::gpu::test
{
template <typename T> DeviceVector<T> upload(const std::vector<T> &host)
{
  return DeviceVector<T>::from_host(host.data(), host.size());
}

/// std::vector<bool> is bit-packed: unpacked to one byte per element first.
inline DeviceVector<bool> upload(const std::vector<bool> &host)
{
  const auto staging = std::make_unique<bool[]>(host.size());
  std::copy(host.begin(), host.end(), staging.get());
  return DeviceVector<bool>::from_host(staging.get(), host.size());
}

/// The `n` elements at `device`.
template <typename T> std::vector<T> download(const T *device, std::size_t n, cudaStream_t stream)
{
  std::vector<T> host(n);
  if (n > 0)
  {
    CHECK_GPU(cudaMemcpyAsync(host.data(), device, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }
  CHECK_GPU(cudaStreamSynchronize(stream));
  return host;
}

/// std::vector<bool> has no contiguous storage to copy into: staged through one byte per element.
inline std::vector<bool> download(const bool *device, std::size_t n, cudaStream_t stream)
{
  const auto staging = std::make_unique<bool[]>(n);
  if (n > 0)
  {
    CHECK_GPU(cudaMemcpyAsync(staging.get(), device, n * sizeof(bool), cudaMemcpyDeviceToHost, stream));
  }
  CHECK_GPU(cudaStreamSynchronize(stream));
  return std::vector<bool>(staging.get(), staging.get() + n);
}

template <typename T> std::vector<T> download(const DeviceVector<T> &device, cudaStream_t stream)
{
  return download(device.data(), device.size(), stream);
}

template <typename T> T download_scalar(const T *device, cudaStream_t stream)
{
  T host{};
  CHECK_GPU(cudaMemcpyAsync(&host, device, sizeof(T), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaStreamSynchronize(stream));
  return host;
}
} // namespace breifast::gpu::test
