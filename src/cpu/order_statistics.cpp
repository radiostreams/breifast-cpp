#include "order_statistics.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <tracy/Tracy.hpp>
#include <xsimd/xsimd.hpp>

namespace breifast::detail::median
{
namespace
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
 * @brief Order-preserving key of a float: `key(a) < key(b)` iff `a < b`. -0 sorts just below +0, NaN beyond +-inf.
 */
struct SignedKey
{
  /// @brief Key of `value`.
  static uint32_t key(float value)
  {
    const auto bits = std::bit_cast<uint32_t>(value);
    const auto mask = static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000U;
    return bits ^ mask;
  }
  /// @brief Keys of a batch, lane by lane the same as the scalar key().
  static xsimd::batch<uint32_t> key(xsimd::batch<float> value)
  {
    const auto bits = xsimd::bitwise_cast<int32_t>(value);
    const auto mask = xsimd::bitwise_or(bits >> 31, xsimd::batch<int32_t>(0x80000000U));
    return xsimd::bitwise_cast<uint32_t>(bits ^ mask);
  }
  /// @brief Inverse of key(): the float a key was made from.
  static float value(uint32_t key)
  {
    return std::bit_cast<float>(key ^ ((key & 0x80000000U) ? 0x80000000U : 0xFFFFFFFFU));
  }
};

/**
 * @brief Key of `|value|`: the bits of a non-negative float already sort like its value.
 */
struct AbsKey
{
  /// @brief Key of `|value|`.
  static uint32_t key(float value)
  {
    return std::bit_cast<uint32_t>(value) & 0x7FFFFFFFU;
  }
  /// @brief Keys of a batch, lane by lane the same as the scalar key().
  static xsimd::batch<uint32_t> key(xsimd::batch<float> value)
  {
    return xsimd::bitwise_and(xsimd::bitwise_cast<uint32_t>(value), xsimd::batch<uint32_t>(0x7FFFFFFFU));
  }
  /// @brief Inverse of key(): `|value|`.
  static float value(uint32_t key)
  {
    return std::bit_cast<float>(key);
  }
};

/**
 * @brief Exact median of the keys of `values`, reading `values` twice and never writing to it.
 *
 * Pass 1 histograms the top 16 key bits to find the bin holding rank k = n/2. Pass 2 histograms the low 16 bits of
 * the keys in that bin, which pins rank k exactly. For even n the median averages ranks k-1 and k.
 * Both passes compute keys on xsimd batches; histogram increments stay scalar, and a scalar loop covers the values past
 * the last whole batch.
 * @tparam Key SignedKey for the median of `values`, AbsKey for the median of `|values|`.
 * @param values Input, left unmodified.
 * @return Median as numpy computes it, NaN for empty input.
 */
template <typename Key> float select(std::span<const float> values)
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

  size_t i;
  for (i = 0; i + xsimd::batch<float>::size <= n; i += xsimd::batch<float>::size)
  {
    const auto batch = xsimd::load_unaligned(&values[i]);
    const auto keys = Key::key(batch);
    alignas(64) uint32_t highs[xsimd::batch<float>::size];
    (keys >> 16).store_aligned(highs);
    for (size_t j = 0; j < xsimd::batch<float>::size; ++j)
    {
      histogram[highs[j]]++;
    }
  }
  for (; i < n; ++i)
  {
    histogram[Key::key(values[i]) >> 16]++;
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

  xsimd::batch<uint32_t> lower_max_vec(0);
  const xsimd::batch<uint32_t> bin_vec(bin);
  const xsimd::batch<uint32_t> zero(0);
  for (i = 0; i + xsimd::batch<float>::size <= n; i += xsimd::batch<float>::size)
  {
    const auto batch = xsimd::load_unaligned(&values[i]);
    const auto keys = Key::key(batch);

    const xsimd::batch<uint32_t> high = keys >> 16;

    lower_max_vec = xsimd::max(lower_max_vec, xsimd::select(high < bin_vec, keys, zero));

    if ((high == bin_vec).mask())
    {
      alignas(64) uint32_t keys_arr[xsimd::batch<float>::size];
      (keys).store_aligned(keys_arr);
      alignas(64) uint32_t highs[xsimd::batch<float>::size];
      (high).store_aligned(highs);
      for (size_t j = 0; j < xsimd::batch<float>::size; ++j)
      {
        if (highs[j] == bin)
        {
          histogram[keys_arr[j] & 0xFFFFU]++;
        }
      }
    }
  }

  auto lower_max_key = xsimd::reduce_max(lower_max_vec);
  for (; i < n; ++i)
  {
    const uint32_t key = Key::key(values[i]);
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

  const float hi = Key::value((bin << 16) | k_low);
  if (n % 2 == 1)
  {
    return hi;
  }
  const float lo = need_k_minus_1 ? Key::value((bin << 16) | k_minus_1_low) : Key::value(lower_max_key);
  return (lo + hi) / 2.F;
}
} // namespace radix
} // namespace

float radix_median_of(std::span<const float> values)
{
  ZoneScoped;
  return radix::select<radix::SignedKey>(values);
}

float radix_median_of_abs(std::span<const float> values)
{
  ZoneScoped;
  return radix::select<radix::AbsKey>(values);
}

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

template float median_of_abs<float>(std::span<const float>, std::vector<float> &);
template float median_of_abs<double>(std::span<const double>, std::vector<double> &);
template float median_of<float>(std::span<const float>, std::vector<float> &);
template float median_of<double>(std::span<const double>, std::vector<double> &);
} // namespace breifast::detail::median
