#pragma once
#include <cstddef>
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

/// Bins of one radix-select digit: keys are selected one byte at a time, most significant first.
inline constexpr uint32_t radix_bins = 256;

/// One digit of a radix select. `histogram` counts the current digit (at `shift`) of the keys that match `prefix` on
/// `coverage`; `below` counts the keys ordered before that prefix. Finds the bin that holds rank `rank`, advances
/// `below` past the bins before it, and extends `prefix` and `coverage` with it.
__device__ inline void select_radix_digit(const uint32_t *histogram, const size_t rank, const int shift,
                                          uint32_t &below, uint32_t &prefix, uint32_t &coverage)
{
  size_t cumulative = below;
  uint32_t bin = 0;
  for (; bin < radix_bins; ++bin)
  {
    if (cumulative + histogram[bin] > rank)
    {
      break;
    }
    cumulative += histogram[bin];
  }
  below = static_cast<uint32_t>(cumulative);
  prefix |= bin << shift;
  coverage |= 0xFFU << shift;
}

/// Median of `n` keys from `hi_key`, the key of rank n / 2, and `below`, the count of keys strictly smaller than it.
/// Odd n: hi_key's value. Even n: averaged with rank n / 2 - 1, which is hi_key again when it repeats below rank n / 2
/// (below < n / 2), else `lower_max`, the largest key smaller than hi_key.
__device__ inline float median_of_keys(const size_t n, const uint32_t below, const uint32_t hi_key,
                                       const uint32_t lower_max, const bool abs_mode)
{
  const auto value = [abs_mode](const uint32_t key) { return abs_mode ? abs_value(key) : signed_value(key); };
  const float hi = value(hi_key);
  if (n % 2 == 1)
  {
    return hi;
  }
  const uint32_t lo_key = n / 2 > below ? hi_key : lower_max;
  return (value(lo_key) + hi) / 2.F;
}
} // namespace breifast::gpu
