#pragma once
#include <cstdint>
#include <cstring>

namespace breifast::gpu
{
/// Order-preserving key of a float: `key(a) < key(b)` iff `a < b`. -0 sorts just below +0, NaN beyond +-inf. Twin of
/// SignedKey in src/cpu/order_statistics.cpp.
__host__ __device__ inline uint32_t signed_key(float value)
{
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits ^ (static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000U);
}

/// Inverse of signed_key().
__host__ __device__ inline float signed_value(uint32_t key)
{
  const uint32_t bits = key ^ ((key & 0x80000000U) ? 0x80000000U : 0xFFFFFFFFU);
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

/// Key of `|value|`: the bits of a non-negative float already sort like its value. Twin of AbsKey.
__host__ __device__ inline uint32_t abs_key(float value)
{
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits & 0x7FFFFFFFU;
}

/// Inverse of abs_key().
__host__ __device__ inline float abs_value(uint32_t key)
{
  float value;
  memcpy(&value, &key, sizeof(value));
  return value;
}
} // namespace breifast::gpu
