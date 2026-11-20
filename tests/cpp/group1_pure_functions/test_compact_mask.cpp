#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "cpu/windows.hpp"

using breifast::plane_lot::detail::compact_mask;
using Catch::Matchers::Equals;

namespace
{
// The mask is a byte per value (0 or 1), reinterpreted as bool for the span.
std::vector<size_t> run_compact(const std::vector<unsigned char> &mask)
{
  std::vector<size_t> out(mask.size());
  const auto *flags = reinterpret_cast<const bool *>(mask.data());
  const size_t count = compact_mask({flags, mask.size()}, {out.data(), out.size()});
  out.resize(count);
  return out;
}
} // namespace

TEST_CASE("compact_mask returns the true indices in raster order", "[compact_mask]")
{
  // numpy: np.flatnonzero([1, 0, 1, 1, 0, 0, 1, 0]) = [0, 2, 3, 6].
  CHECK_THAT(run_compact({1, 0, 1, 1, 0, 0, 1, 0}), Equals(std::vector<size_t>{0, 2, 3, 6}));

  // Every other pixel.
  CHECK_THAT(run_compact({1, 0, 1, 0, 1, 0}), Equals(std::vector<size_t>{0, 2, 4}));
}

TEST_CASE("compact_mask handles all-false, all-true, single and empty masks", "[compact_mask]")
{
  CHECK_THAT(run_compact(std::vector<unsigned char>(5, 0)), Equals(std::vector<size_t>{}));
  CHECK_THAT(run_compact(std::vector<unsigned char>(4, 1)), Equals(std::vector<size_t>{0, 1, 2, 3}));

  CHECK_THAT(run_compact({1, 0, 0, 0}), Equals(std::vector<size_t>{0}));
  CHECK_THAT(run_compact({0, 0, 0, 1}), Equals(std::vector<size_t>{3}));

  CHECK_THAT(run_compact({}), Equals(std::vector<size_t>{}));
}

TEST_CASE("compact_mask matches the true indices of a seeded pattern", "[compact_mask]")
{
  // A deterministic pattern; the reference is the list of indices where the byte is nonzero.
  std::vector<unsigned char> mask(257);
  std::vector<size_t> expected;
  for (size_t i = 0; i < mask.size(); ++i)
  {
    mask[i] = static_cast<unsigned char>((i * 7 + 3) % 5 < 2);
    if (mask[i] != 0)
    {
      expected.push_back(i);
    }
  }

  CHECK_THAT(run_compact(mask), Equals(expected));
}
