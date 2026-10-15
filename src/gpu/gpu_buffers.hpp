#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>

namespace breifast::gpu
{
template <typename T> using DeviceVector = thrust::device_vector<T>;

/// Host vector in page-locked memory, the source and destination of asynchronous copies.
template <typename T>
using PinnedVector = thrust::host_vector<
    T, thrust::mr::stateless_resource_allocator<T, thrust::cuda::universal_host_pinned_memory_resource>>;
} // namespace breifast::gpu
