#pragma once

#include <cuda_runtime.h>

namespace breifast::gpu
{
template <typename T> __device__ T block_sum(T value, T *shared)
{
  __syncthreads();
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (static_cast<int>(threadIdx.x) < stride)
    {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  __syncthreads();
  return shared[0];
}

__device__ inline float block_min(float value, float *shared)
{
  __syncthreads();
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (static_cast<int>(threadIdx.x) < stride)
    {
      shared[threadIdx.x] = fminf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  __syncthreads();
  return shared[0];
}

__device__ inline float block_max(float value, float *shared)
{
  __syncthreads();
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (static_cast<int>(threadIdx.x) < stride)
    {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  __syncthreads();
  return shared[0];
}
} // namespace breifast::gpu
