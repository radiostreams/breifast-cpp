#pragma once
#include <cstddef>

namespace breifast::plane_lot::detail
{

/// True when any of the `n` values is true. Scans in blocks of 4096 so it stops at the first block that has one.
bool any_true(const bool *values, size_t n);

} // namespace breifast::plane_lot::detail
