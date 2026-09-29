/**
 * @file median.hpp
 * @brief Exact medians over float arrays.
 *
 * radix_median_of()/radix_median_of_abs() are used in production: two read-only passes, no copy of the input.
 * median_of()/median_of_abs() are the nth_element versions, kept as the reference for the tests.
 */
#pragma once
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <tracy/Tracy.hpp>

namespace breifast::detail::median
{
namespace radix
{
/// @brief Histogram bins per pass: each pass resolves 16 of the 32 key bits.
constexpr size_t n_bins = 1 << 16;

/**
 * @brief Per-thread histogram (256 KB), reused by both passes and both key kinds.
 * @return This thread's histogram, `n_bins` entries.
 */
inline std::vector<uint32_t> &histogram_scratch()
{
  static thread_local std::vector<uint32_t> histogram(n_bins);
  return histogram;
}

/**
 * @brief Order-preserving map: `signed_key(a) < signed_key(b)` iff `a < b`.
 *
 * -0 sorts just below +0, NaN beyond +-inf.
 * @param value Any float.
 * @return Key of `value`.
 */
inline uint32_t signed_key(float value)
{
  const auto bits = std::bit_cast<uint32_t>(value);
  const auto mask = static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000U;
  return bits ^ mask;
}

/**
 * @brief Inverse of signed_key().
 * @param key Key produced by signed_key().
 * @return The original float.
 */
inline float from_signed_key(uint32_t key)
{
  return std::bit_cast<float>(key ^ ((key & 0x80000000U) ? 0x80000000U : 0xFFFFFFFFU));
}

/**
 * @brief Key of `|value|`: the bits of a non-negative float already sort like its value.
 * @param value Any float.
 * @return Bits of `|value|`.
 */
inline uint32_t abs_key(float value)
{
  return std::bit_cast<uint32_t>(value) & 0x7FFFFFFFU;
}

/**
 * @brief Inverse of abs_key().
 * @param key Key produced by abs_key().
 * @return `|value|`.
 */
inline float from_abs_key(uint32_t key)
{
  return std::bit_cast<float>(key);
}

/**
 * @brief Exact median of the keys of `values`, reading `values` twice and never writing to it.
 *
 * Pass 1 histograms the top 16 key bits to find the bin holding rank k = n/2. Pass 2 histograms the low 16 bits of
 * the keys in that bin, which pins rank k exactly. For even n the median averages ranks k-1 and k.
 * @param values Input, left unmodified.
 * @param to_key Order-preserving float -> uint32 map.
 * @param from_key Inverse of `to_key`.
 * @return Median as numpy computes it, NaN for empty input.
 */
template <typename ToKey, typename FromKey> float select(std::span<const float> values, ToKey to_key, FromKey from_key)
{
  const size_t n = values.size();
  if (n == 0)
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const size_t k = n / 2;
  auto &histogram = histogram_scratch();

  // Pass 1: the bin holding rank k, and how many keys sort below it.
  std::fill(histogram.begin(), histogram.end(), 0U);
  for (const float value : values)
  {
    histogram[to_key(value) >> 16]++;
  }

  uint32_t bin = 0;
  size_t below = 0;
  while (below + histogram[bin] <= k)
  {
    below += histogram[bin];
    ++bin;
  }

  // Pass 2: low bits of the keys in that bin, plus the largest key below it (rank k-1 when k == below).
  std::fill(histogram.begin(), histogram.end(), 0U);
  uint32_t lower_max_key = 0;
  for (const float value : values)
  {
    const uint32_t key = to_key(value);
    const uint32_t high = key >> 16;
    if (high == bin)
    {
      histogram[key & 0xFFFFU]++;
    }
    else if (high < bin)
    {
      lower_max_key = std::max(lower_max_key, key);
    }
  }

  const bool need_k_minus_1 = (n % 2 == 0) && (k - 1 >= below);
  bool found_k_minus_1 = !need_k_minus_1;
  uint32_t k_low = 0;
  uint32_t k_minus_1_low = 0;
  size_t cumulative = below;
  for (uint32_t low = 0;; ++low)
  {
    cumulative += histogram[low];
    if (!found_k_minus_1 && cumulative > k - 1)
    {
      k_minus_1_low = low;
      found_k_minus_1 = true;
    }
    if (cumulative > k)
    {
      k_low = low;
      break;
    }
  }

  const float hi = from_key((bin << 16) | k_low);
  if (n % 2 == 1)
  {
    return hi;
  }
  const float lo = need_k_minus_1 ? from_key((bin << 16) | k_minus_1_low) : from_key(lower_max_key);
  return (lo + hi) / 2.F;
}
} // namespace radix

/**
 * @brief Exact median of `values`, same result as median_of().
 * @param values Input, left unmodified.
 * @return Median, NaN for empty input.
 */
inline float radix_median_of(std::span<const float> values)
{
  ZoneScoped;
  return radix::select(values, radix::signed_key, radix::from_signed_key);
}

/**
 * @brief Exact median of `|values|`, same result as median_of_abs().
 * @param values Input, left unmodified.
 * @return Median of the absolute values, NaN for empty input.
 */
inline float radix_median_of_abs(std::span<const float> values)
{
  ZoneScoped;
  return radix::select(values, radix::abs_key, radix::from_abs_key);
}
} // namespace breifast::detail::median

namespace breifast::detail::median
{
/**
 * @brief Reference median of `|values|`: copies `|values|` into `scratch`, then std::nth_element.
 * @param values Input, left unmodified.
 * @param scratch Buffer resized to `values.size()` and overwritten.
 * @return Median of the absolute values.
 */
template <typename T> float median_of_abs(std::span<const T> values, std::vector<T> &scratch)
{
  ZoneScoped;
  scratch.resize(values.size());
  std::transform(values.begin(), values.end(), scratch.begin(), [](T x) { return std::abs(x); });
  const size_t n = scratch.size();
  const size_t mid = n / 2;
  const auto cmp = [](T a, T b) { return a < b; };

  std::nth_element(scratch.begin(), scratch.begin() + mid, scratch.end(), cmp);
  const T hi = scratch[mid];
  if (n % 2 == 1)
  {
    return hi;
  }

  T lo = scratch[0];
  for (size_t i = 1; i < mid; ++i)
  {
    lo = std::max(lo, scratch[i]);
  }
  return (hi + lo) / 2.F;
}

/**
 * @brief Reference median of `values`: copies `values` into `scratch`, then std::nth_element.
 * @param values Input, left unmodified.
 * @param scratch Buffer resized to `values.size()` and overwritten.
 * @return Median.
 */
template <typename T> float median_of(std::span<const T> values, std::vector<T> &scratch)
{
  ZoneScoped;
  scratch.assign(values.begin(), values.end());
  const size_t n = scratch.size();
  const size_t mid = n / 2;
  const auto cmp = [](T a, T b) { return a < b; };

  std::nth_element(scratch.begin(), scratch.begin() + mid, scratch.end(), cmp);
  const T hi = scratch[mid];
  if (n % 2 == 1)
  {
    return hi;
  }

  T lo = scratch[0];
  for (size_t i = 1; i < mid; ++i)
  {
    lo = std::max(lo, scratch[i]);
  }
  return (hi + lo) / 2.F;
}
} // namespace breifast::detail::median