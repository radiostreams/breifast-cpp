#include "scalar_reductions.hpp"

#include <algorithm>

namespace breifast::plane_lot::detail
{

bool any_true(const bool *values, size_t n)
{
  constexpr size_t block_size = 4096;
  for (size_t block_start = 0; block_start < n; block_start += block_size)
  {
    const size_t block_end = std::min(block_start + block_size, n);
    unsigned char accumulated = 0;
    for (size_t i = block_start; i < block_end; ++i)
    {
      accumulated |= static_cast<unsigned char>(values[i]);
    }
    if (accumulated != 0)
    {
      return true;
    }
  }
  return false;
}

} // namespace breifast::plane_lot::detail
