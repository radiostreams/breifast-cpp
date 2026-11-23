#include <algorithm>
#include <cassert>

#include <cub/warp/warp_scan.cuh>

#include "gpu_kernels.hpp"

#include "gpu_device_utils.cuh"
#include "radix_select.cuh"

namespace breifast::gpu
{

namespace
{
__device__ inline long min_of(long i, long radius)
{
  const long bound = i - radius;
  return bound > 0 ? bound : 0;
}

__device__ inline long max_of(long i, long window_size, long radius, long n)
{
  const long bound = i + window_size - radius;
  return bound < n ? bound : n;
}

/// Warps of a tile_prefix block: each takes 32 / tile_warps rows of the tile, then as many columns.
constexpr unsigned int tile_warps = 8;

/// Step 1 of a part of the RMS map: the 2D prefix of plane^2 inside each 32 x 32 tile of the part's region, into
/// part_prefix. A block of 32 x tile_warps threads takes one tile: it loads it coalesced, scans its rows then its
/// columns in shared memory, and writes it back coalesced. Launch with a grid of tiles_x x tiles_y blocks of
/// dim3(32, tile_warps).
__global__ void tile_prefix(const float *plane, long n_cols,    // the image, row-major
                            long region_y0, long region_x0,     // the part's region, in image coordinates
                            long region_rows, long region_cols, // already clipped to the image
                            double *part_prefix,                // region_rows x region_cols, row-major
                            double *last_cols)                  // tiles_x x region_rows: each tile's last column
{
  __shared__ double tile[32][33]; // padding to avoid bank sharing issues

  const auto lane = threadIdx.x;
  const auto warp = threadIdx.y;

  const long tile_x0 = blockIdx.x * 32;
  const long tile_y0 = blockIdx.y * 32;
  const long col = tile_x0 + lane;

#pragma unroll
  for (unsigned int local_row = warp; local_row < 32; local_row += tile_warps)
  {
    double value = 0.;
    const long row = tile_y0 + local_row;
    if (row < region_rows && col < region_cols)
    {
      const float pixel = plane[(region_y0 + row) * n_cols + region_x0 + col];
      value = static_cast<double>(pixel) * pixel;
    }
    tile[local_row][lane] = value;
  }
  __syncthreads();

  using WarpScan = cub::WarpScan<double>;
  __shared__ typename WarpScan::TempStorage scan_storage[tile_warps];

#pragma unroll
  for (unsigned int local_row = warp; local_row < 32; local_row += tile_warps)
  {
    double sum;
    const auto value = tile[local_row][lane];
    WarpScan(scan_storage[warp]).InclusiveSum(value, sum);
    __syncwarp();
    tile[local_row][lane] = sum;
  }

  __syncthreads();

#pragma unroll
  for (unsigned int local_col = warp; local_col < 32; local_col += tile_warps)
  {
    double sum;
    const auto value = tile[lane][local_col];
    WarpScan(scan_storage[warp]).InclusiveSum(value, sum);
    __syncwarp();
    tile[lane][local_col] = sum;
  }

  __syncthreads();

#pragma unroll
  for (unsigned int local_row = warp; local_row < 32; local_row += tile_warps)
  {
    const long row = tile_y0 + local_row;
    if (row < region_rows && col < region_cols)
    {
      part_prefix[row * region_cols + col] = tile[local_row][lane];
    }
  }
  // The tile's last column again, along y, so that part_carries reads it coalesced.
  if (warp == 0 && tile_y0 + lane < region_rows)
  {
    last_cols[blockIdx.x * region_rows + tile_y0 + lane] = tile[lane][31];
  }
}

/// What each 32 x 32 tile of the part adds to its values to turn them into the prefix of the whole part. For a pixel
/// (y, x) of tile (tile_row, tile_col):
///
///   S(y, x) = part_prefix[y][x] + above[tile_row][x] + left[tile_col][y] + corner[tile_row][tile_col]
///
///   above:  the tiles directly above, columns up to x: the sum of their last rows at column x
///   left:   the tiles directly to the left, rows up to y: the sum of their last columns at row y
///   corner: the tiles above and to the left: the sum of their totals
///
/// All three in one launch, part_carries. The tiles counted are always before another tile, so they are whole: the last
/// tile of a line counts 0.
constexpr long carry_warps = 8;           // warps of a part_carries block
constexpr long carry_tiles_per_warp = 8;  // tile slots of a thread in an above or left block
constexpr long max_carry_tiles = carry_warps * carry_tiles_per_warp; // 64: regions of up to 2048 x 2048
static_assert(carry_warps * 32 == block_size);

/// Exclusive sums down the tiles of 32 neighbouring lines, lane l taking line first_line + l: in[tile * in_stride +
/// line] is the line's value at a tile, and out[tile * out_stride + line] receives the sum of its values at the tiles
/// before. Warp w takes tile slots 8 w to 8 w + 7. The neighbouring lines of a warp are neighbouring addresses, so the
/// loads and stores are coalesced. Lanes past the last line and slots past the last tile load a valid clamped address
/// and count 0, without branching.
__device__ void carry_lines(const double *in, const long in_stride, double *out, const long out_stride,
                            const long first_line, const long n_lines, const long n_tiles)
{
  __shared__ double warp_totals[carry_warps][32];
  const long lane = threadIdx.x % 32;
  const long warp = threadIdx.x / 32;
  const long line = first_line + lane;
  const bool line_inside = line < n_lines;
  const long loaded_line = line_inside ? line : n_lines - 1;
  const long first_tile = warp * carry_tiles_per_warp;

  double values[carry_tiles_per_warp];
  double warp_total = 0.;
#pragma unroll
  for (long k = 0; k < carry_tiles_per_warp; ++k)
  {
    const long tile = first_tile + k;
    const bool counted = line_inside && tile < n_tiles - 1;
    const double loaded = in[(counted ? tile : 0) * in_stride + loaded_line];
    values[k] = counted ? loaded : 0.;
    warp_total += values[k];
  }
  warp_totals[warp][lane] = warp_total;
  __syncthreads();

  double before = 0.; // the line's values at the tiles of the warps before this one
#pragma unroll
  for (long other_warp = 0; other_warp < carry_warps; ++other_warp)
  {
    before += other_warp < warp ? warp_totals[other_warp][lane] : 0.;
  }
#pragma unroll
  for (long k = 0; k < carry_tiles_per_warp; ++k)
  {
    const long tile = first_tile + k;
    if (line_inside && tile < n_tiles)
    {
      out[tile * out_stride + line] = before;
    }
    before += values[k];
  }
}

/// The exclusive sums of a warp's 64 values, lane l holding values 2 l and 2 l + 1: sums[k] is the sum of all the values
/// before values[k].
__device__ void warp_exclusive_sums(const double (&values)[2], double (&sums)[2])
{
  using WarpScan = cub::WarpScan<double>;
  __shared__ typename WarpScan::TempStorage scan_storage[carry_warps];
  double before = 0.;
  WarpScan(scan_storage[threadIdx.x / 32]).ExclusiveSum(values[0] + values[1], before);
  sums[0] = before;
  sums[1] = before + values[0];
}

/// corner: the 2D exclusive prefix of the tile totals, in shared memory: down the tile rows, then along the tile columns.
/// A tile's total is its last column at its last row. Each warp takes 8 columns, then 8 rows, 2 values per lane.
__device__ void carry_corners(const double *last_cols, const long region_rows, const long tiles_y, const long tiles_x,
                              double *corner)
{
  __shared__ double totals[max_carry_tiles][max_carry_tiles + 1]; // + 1: the column reads hit different banks
  const long lane = threadIdx.x % 32;
  const long warp = threadIdx.x / 32;

  for (long index = threadIdx.x; index < max_carry_tiles * max_carry_tiles; index += block_size)
  {
    const long tile_row = index / max_carry_tiles;
    const long tile_col = index % max_carry_tiles;
    const bool counted = tile_row < tiles_y - 1 && tile_col < tiles_x - 1;
    const double loaded = last_cols[(counted ? tile_col * region_rows + 32 * tile_row + 31 : 0)];
    totals[tile_row][tile_col] = counted ? loaded : 0.;
  }
  __syncthreads();

  for (long tile_col = warp; tile_col < max_carry_tiles; tile_col += carry_warps) // down the tile rows
  {
    const double values[2] = {totals[2 * lane][tile_col], totals[2 * lane + 1][tile_col]};
    double sums[2];
    warp_exclusive_sums(values, sums);
    totals[2 * lane][tile_col] = sums[0];
    totals[2 * lane + 1][tile_col] = sums[1];
  }
  __syncthreads();

  for (long tile_row = warp; tile_row < max_carry_tiles; tile_row += carry_warps) // along the tile columns
  {
    const double values[2] = {totals[tile_row][2 * lane], totals[tile_row][2 * lane + 1]};
    double sums[2];
    warp_exclusive_sums(values, sums);
#pragma unroll
    for (long k = 0; k < 2; ++k)
    {
      const long tile_col = 2 * lane + k;
      if (tile_row < tiles_y && tile_col < tiles_x)
      {
        corner[tile_row * tiles_x + tile_col] = sums[k];
      }
    }
  }
}

/// Step 2 of a part: above, left and corner in one launch. Blocks 0 .. above_blocks - 1 take 32 columns each (above),
/// the next left_blocks 32 rows each (left), the last one the corners. The role is the same for a whole block. Launch
/// with carry_blocks() blocks of block_size.
__global__ void part_carries(const double *part_prefix, const double *last_cols, long region_rows, long region_cols,
                             double *above,  // tiles_y x region_cols
                             double *left,   // tiles_x x region_rows
                             double *corner) // tiles_y x tiles_x
{
  const long tiles_y = (region_rows + 31) / 32;
  const long tiles_x = (region_cols + 31) / 32;
  const long above_blocks = (region_cols + 31) / 32;
  const long left_blocks = (region_rows + 31) / 32;
  const long block = blockIdx.x;

  if (block < above_blocks)
  {
    // A tile row's last row at column x: part_prefix[(32 tile_row + 31) * region_cols + x].
    carry_lines(part_prefix + 31 * region_cols, 32 * region_cols, above, region_cols, block * 32, region_cols, tiles_y);
  }
  else if (block < above_blocks + left_blocks)
  {
    carry_lines(last_cols, region_rows, left, region_rows, (block - above_blocks) * 32, region_rows, tiles_x);
  }
  else
  {
    carry_corners(last_cols, region_rows, tiles_y, tiles_x, corner);
  }
}

/// Blocks of part_carries for a region.
unsigned int carry_blocks(const long region_rows, const long region_cols)
{
  return static_cast<unsigned int>((region_cols + 31) / 32 + (region_rows + 31) / 32 + 1);
}

/// Step 3 of a part of the RMS map: each output's RMS from its box sum, 4 values of the part's prefix (part_prefix plus
/// above, left and corner). One thread per output of the part.
__global__ void box_rms(const double *part_prefix, const double *above, const double *left, const double *corner,
                        long region_y0, long region_x0, long region_rows, long region_cols, // the part's region
                        long part_y0, long part_x0, long part_rows, long part_cols,         // its outputs
                        long n_rows, long n_cols, long box_size,                            // the image
                        float *rms)                                                         // n_rows x n_cols
{
  const auto index = static_cast<long>(global_index());
  if (index >= part_rows * part_cols)
  {
    return;
  }
  const long y = part_y0 + index / part_cols;
  const long x = part_x0 + index % part_cols;

  const long radius = box_size / 2;
  const long window_size = box_size > 0 ? box_size : 1;
  const long row_min = min_of(y, radius);
  const long row_max = max_of(y, window_size, radius, n_rows) - 1;
  const long col_min = min_of(x, radius);
  const long col_max = max_of(x, window_size, radius, n_cols) - 1;

  // The prefix of the whole part at a position of the region, 0 above or left of it.
  const long tiles_x = (region_cols + 31) / 32;
  const auto prefix_at = [&](const long region_y, const long region_x)
  {
    if (region_y < 0 || region_x < 0)
    {
      return 0.;
    }
    return part_prefix[region_y * region_cols + region_x] + above[(region_y / 32) * region_cols + region_x] +
           left[(region_x / 32) * region_rows + region_y] + corner[(region_y / 32) * tiles_x + region_x / 32];
  };

  // The box's last row and column, and the row and column just before it, in region coordinates.
  const long last_row = row_max - region_y0;
  const long row_before = row_min - 1 - region_y0;
  const long last_col = col_max - region_x0;
  const long col_before = col_min - 1 - region_x0;
  const double box_sum = prefix_at(last_row, last_col) - prefix_at(row_before, last_col) -
                         prefix_at(last_row, col_before) + prefix_at(row_before, col_before);

  const auto count = static_cast<double>((row_max - row_min + 1) * (col_max - col_min + 1));
  rms[y * n_cols + x] = static_cast<float>(sqrt(box_sum / count));
}

// The base noise is max(rms[p], *base_noise) with an RMS map, else *base_noise. Stores each pixel's noise as the
// median's first pass reads it.
struct CombineNoise
{
  const float *rms;
  const float *base_noise;
  const float *selfnoise_offset;
  const float *dde_noise;
  float selfnoise_scale;
  float dde_scale;
  float *out_noise_map;

  __device__ float noise(const float rms_value, const float offset, const float dde) const
  {
    const float base = rms != nullptr ? fmaxf(rms_value, *base_noise) : *base_noise;
    return sqrtf(base * base + selfnoise_scale * offset + dde_scale * dde);
  }
  __device__ float operator()(const size_t i) const
  {
    const float value = noise(rms != nullptr ? rms[i] : 0.F, selfnoise_offset[i], dde_noise[i]);
    out_noise_map[i] = value;
    return value;
  }
  __device__ float4 load4(const size_t i4) const
  {
    const float4 r = rms != nullptr ? reinterpret_cast<const float4 *>(rms)[i4] : float4{};
    const float4 o = reinterpret_cast<const float4 *>(selfnoise_offset)[i4];
    const float4 d = reinterpret_cast<const float4 *>(dde_noise)[i4];
    const float4 value = {noise(r.x, o.x, d.x), noise(r.y, o.y, d.y), noise(r.z, o.z, d.z), noise(r.w, o.w, d.w)};
    reinterpret_cast<float4 *>(out_noise_map)[i4] = value;
    return value;
  }
};

/// Outputs per side of a part of the RMS map. A part's prefix covers its outputs plus the reach of the box (1935 x 1935
/// doubles, 30 MB, with a box of 400) and stays in L2 from its tile scans to its box sums.
constexpr long rms_part_size = 1536;

/// Where gpu_rms_noise_map's scratch holds each array, sized for the largest part's region: part_prefix, then above,
/// left, corner and last_cols.
struct RmsScratchLayout
{
  long max_rows;
  long max_cols;
  long max_tiles_y;
  long max_tiles_x;

  RmsScratchLayout(const std::size_t n_y, const std::size_t n_x, const long boxsize)
  {
    const long window_size = boxsize > 0 ? boxsize : 1;
    max_rows = std::min(rms_part_size + window_size - 1, static_cast<long>(n_y));
    max_cols = std::min(rms_part_size + window_size - 1, static_cast<long>(n_x));
    max_tiles_y = (max_rows + 31) / 32;
    max_tiles_x = (max_cols + 31) / 32;
  }
  std::size_t prefix_size() const { return static_cast<std::size_t>(max_rows * max_cols); }
  std::size_t above_size() const { return static_cast<std::size_t>(max_tiles_y * max_cols); }
  std::size_t left_size() const { return static_cast<std::size_t>(max_tiles_x * max_rows); } // also last_cols
  std::size_t corner_size() const { return static_cast<std::size_t>(max_tiles_y * max_tiles_x); }
  std::size_t total() const { return prefix_size() + above_size() + 2 * left_size() + corner_size(); }
};
} // namespace

std::size_t gpu_rms_noise_map_scratch_size(const std::size_t n_y, const std::size_t n_x, const long boxsize)
{
  return RmsScratchLayout(n_y, n_x, boxsize).total();
}

void gpu_rms_noise_map(const float *plane, size_t n_y, size_t n_x, long boxsize, double *scratch, float *out_rms,
                       float *out_median, MedianScratch &median_scratch, cudaStream_t stream)
{
  assert(n_y > 0 && n_x > 0);
  const RmsScratchLayout layout(n_y, n_x, boxsize);
  assert(layout.max_tiles_y <= max_carry_tiles && layout.max_tiles_x <= max_carry_tiles); // boxes up to 1025

  double *part_prefix = scratch;
  double *above = part_prefix + layout.prefix_size();
  double *left = above + layout.above_size();
  double *corner = left + layout.left_size();
  double *last_cols = corner + layout.corner_size();

  const auto rows = static_cast<long>(n_y);
  const auto cols = static_cast<long>(n_x);
  const long radius = boxsize / 2;
  const long window_size = boxsize > 0 ? boxsize : 1;

  // The parts one after the other on the stream, sharing the scratch. A part's region is its outputs plus the rows and
  // columns their boxes reach, clipped to the image.
  for (long part_y0 = 0; part_y0 < rows; part_y0 += rms_part_size)
  {
    const long part_rows = std::min(rms_part_size, rows - part_y0);
    const long region_y0 = std::max(part_y0 - radius, 0L);
    const long region_rows = std::min(part_y0 + part_rows + window_size - radius - 1, rows) - region_y0;
    for (long part_x0 = 0; part_x0 < cols; part_x0 += rms_part_size)
    {
      const long part_cols = std::min(rms_part_size, cols - part_x0);
      const long region_x0 = std::max(part_x0 - radius, 0L);
      const long region_cols = std::min(part_x0 + part_cols + window_size - radius - 1, cols) - region_x0;
      const long tiles_y = (region_rows + 31) / 32;
      const long tiles_x = (region_cols + 31) / 32;

      const dim3 tile_grid(static_cast<unsigned int>(tiles_x), static_cast<unsigned int>(tiles_y));
      tile_prefix<<<tile_grid, dim3(32, tile_warps), 0, stream>>>(plane, cols, region_y0, region_x0, region_rows,
                                                                   region_cols, part_prefix, last_cols);
      CHECK_GPU(cudaGetLastError());
      launch(part_carries, carry_blocks(region_rows, region_cols), block_size, stream, part_prefix, last_cols,
             region_rows, region_cols, above, left, corner);
      launch_per_element(box_rms, static_cast<std::size_t>(part_rows * part_cols), stream, part_prefix, above, left,
                         corner, region_y0, region_x0, region_rows, region_cols, part_y0, part_x0, part_rows,
                         part_cols, rows, cols, boxsize, out_rms);
    }
  }
  radix_median(RadixValues{out_rms}, out_rms, n_y * n_x, false, 1.F, out_median, median_scratch, stream);
}

void gpu_combine_noise_map(const float *rms, const float *base_noise, const float *selfnoise_offset,
                           const float *dde_noise, const float selfnoise_scale, const float dde_scale,
                           float *out_noise_map, const size_t n_pixels, float *out_median, MedianScratch &scratch,
                           cudaStream_t stream)
{
  assert(n_pixels > 0);
  radix_median(CombineNoise{rms, base_noise, selfnoise_offset, dde_noise, selfnoise_scale, dde_scale, out_noise_map},
               out_noise_map, n_pixels, false, 1.F, out_median, scratch, stream);
}
} // namespace breifast::gpu