#pragma once
// Reductions to a scalar over pixels (theme 3 of optimization_gpu/kernels.md): any_true. stamp_min_max and donut_std
// join it.
#include <cstddef>

namespace breifast::plane_lot::detail
{

/// True when any of the `n` values is true. Scans in blocks of 4096 so it stops at the first block that has one.
bool any_true(const bool *values, size_t n);

} // namespace breifast::plane_lot::detail
