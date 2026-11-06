#pragma once

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
__device__ inline void atomic_min_float(float *address, float value)
{
  int *address_as_int = reinterpret_cast<int *>(address);
  int old = *address_as_int;
  int assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
  } while (assumed != old);
}

__device__ inline void atomic_max_float(float *address, float value)
{
  int *address_as_int = reinterpret_cast<int *>(address);
  int old = *address_as_int;
  int assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);
}
} // namespace breifast::gpu