/**
 * @file order_statistics.hpp
 * @brief Exact medians over float arrays.
 *
 * radix_median_of()/radix_median_of_abs() are used in production: two read-only passes, no copy of the input.
 * median_of()/median_of_abs() are the nth_element versions, kept as the reference for the tests.
 */
#pragma once
#include <span>
#include <vector>

namespace breifast::detail::median
{
/**
 * @brief Exact median of `values`, same result as median_of().
 * @param values Input, left unmodified.
 * @return Median, NaN for empty input.
 */
float radix_median_of(std::span<const float> values);

/**
 * @brief Exact median of `|values|`, same result as median_of_abs().
 * @param values Input, left unmodified.
 * @return Median of the absolute values, NaN for empty input.
 */
float radix_median_of_abs(std::span<const float> values);

/**
 * @brief Reference median of `|values|`: copies `|values|` into `scratch`, then std::nth_element.
 * @param values Input, left unmodified.
 * @param scratch Buffer resized to `values.size()` and overwritten.
 * @return Median of the absolute values.
 */
template <typename T> float median_of_abs(std::span<const T> values, std::vector<T> &scratch);

/**
 * @brief Reference median of `values`: copies `values` into `scratch`, then std::nth_element.
 * @param values Input, left unmodified.
 * @param scratch Buffer resized to `values.size()` and overwritten.
 * @return Median.
 */
template <typename T> float median_of(std::span<const T> values, std::vector<T> &scratch);

// Defined in order_statistics.cpp for float and double.
extern template float median_of_abs<float>(std::span<const float>, std::vector<float> &);
extern template float median_of_abs<double>(std::span<const double>, std::vector<double> &);
extern template float median_of<float>(std::span<const float>, std::vector<float> &);
extern template float median_of<double>(std::span<const double>, std::vector<double> &);
} // namespace breifast::detail::median
