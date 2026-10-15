#pragma once
#include <cstddef>
#include <span>

namespace breifast::plane_lot::detail
{

/**
 * @brief Local RMS noise estimate: a sliding `boxsize x boxsize` window over `plane`, edge-corrected so border windows
 *        use only their true (in-bounds) pixel count.
 *
 * Two separable passes with double accumulators: running sums of `plane^2` along each row (stored as float in
 * `row_window_sums`), then along each column. Writes `out_rms` completely, and fills `column_accumulators`; both
 * workspaces may hold anything on entry. The caller applies `make_noise_map(method='std')`'s median floor.
 * @param plane `[n_y, n_x]` values, row major.
 * @param row_window_sums Workspace of `n_y * n_x` floats.
 * @param column_accumulators Workspace of `n_x` doubles.
 * @param out_rms `[n_y, n_x]` `sqrt(mean of plane^2 within each pixel's window)`.
 */
void rms_noise_map(std::span<const float> plane, const size_t n_y_size, const size_t n_x_size, const long boxsize,
                   std::span<float> row_window_sums, std::span<double> column_accumulators, std::span<float> out_rms);

} // namespace breifast::plane_lot::detail
