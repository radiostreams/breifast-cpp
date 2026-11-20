#include <cassert>
#include <climits>

#include <cub/device/device_scan.cuh>
#include <cuda/std/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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

/// The square of a value, in double.
struct SquareToDouble
{
  __host__ __device__ double operator()(const float value) const
  {
    return static_cast<double>(value) * value;
  }
};

/// The row of the pixel with a given index, the key of a segmented scan.
struct RowOfPixel
{
  size_t n_x;

  __host__ __device__ size_t operator()(const size_t index) const
  {
    return index / n_x;
  }
};

/// Inclusive prefix sums of the squares of every row of `d_plane`, in double, into `d_prefix`: d_prefix[y * n_x + x] is
/// the sum of d_plane[y][0..x]^2. The scan restarts at each row. `scan_storage` is the scan's temporary storage.
void window_kernel(const float *d_plane, const size_t n_y, const size_t n_x, double *d_prefix,
                   DeviceVector<std::byte> &scan_storage, cudaStream_t stream)
{
  const size_t n = n_y * n_x;
  assert(n > 0 && n <= static_cast<size_t>(INT_MAX));

  const auto squares = thrust::make_transform_iterator(d_plane, SquareToDouble{});
  const auto rows = thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0), RowOfPixel{n_x});

  size_t bytes = 0;
  CHECK_GPU(cub::DeviceScan::InclusiveSumByKey(nullptr, bytes, rows, squares, d_prefix, static_cast<int>(n),
                                               cuda::std::equal_to<>{}, stream));
  scan_storage.ensure(bytes);
  bytes = scan_storage.size();
  CHECK_GPU(cub::DeviceScan::InclusiveSumByKey(scan_storage.data(), bytes, rows, squares, d_prefix, static_cast<int>(n),
                                               cuda::std::equal_to<>{}, stream));
}

/// The value at position k of a column-by-column walk of a [n_y, n_x] plane: line k / n_y is column x, and the position
/// k % n_y within it is row y.
struct ColumnWalkValue
{
  const float *plane;
  size_t n_y;
  size_t n_x;

  __host__ __device__ double operator()(const size_t k) const
  {
    return plane[(k % n_y) * n_x + k / n_y];
  }
};

/// The column of position k of a column-by-column walk, the key of a segmented scan.
struct ColumnOfWalkIndex
{
  size_t n_y;

  __host__ __device__ size_t operator()(const size_t k) const
  {
    return k / n_y;
  }
};

/// Inclusive prefix sums down every column of `d_plane`, in double, into `d_prefix` in transposed layout:
/// d_prefix[x * n_y + y] is the sum of d_plane[0..y][x]. The scan restarts at each column.
void window_kernel_cols(const float *d_plane, const size_t n_y, const size_t n_x, double *d_prefix,
                        DeviceVector<std::byte> &scan_storage, cudaStream_t stream)
{
  const size_t n = n_y * n_x;
  assert(n > 0 && n <= static_cast<size_t>(INT_MAX));

  const auto indices = thrust::make_counting_iterator<size_t>(0);
  const auto values = thrust::make_transform_iterator(indices, ColumnWalkValue{d_plane, n_y, n_x});
  const auto columns = thrust::make_transform_iterator(indices, ColumnOfWalkIndex{n_y});

  size_t bytes = 0;
  CHECK_GPU(cub::DeviceScan::InclusiveSumByKey(nullptr, bytes, columns, values, d_prefix, static_cast<int>(n),
                                               cuda::std::equal_to<>{}, stream));
  scan_storage.ensure(bytes);
  bytes = scan_storage.size();
  CHECK_GPU(cub::DeviceScan::InclusiveSumByKey(scan_storage.data(), bytes, columns, values, d_prefix,
                                               static_cast<int>(n), cuda::std::equal_to<>{}, stream));
}

__global__ void window_diff(const double *in, const size_t n_y, const size_t n_x, long box_size, float* out)
{
  const auto i = global_index();
  if (i < n_x * n_y)
  {
    const auto y = i / n_x;
    const auto x = i % n_x;
    const auto row_offset = y * n_x;
    const long radius = box_size / 2;
    const long window_size = box_size > 0 ? box_size : 1;
    const long lo = min_of(x, radius);
    const long hi = max_of(x, window_size, radius, n_x);
    const auto sum = in[row_offset + hi - 1] - (lo > 0 ? in[row_offset + lo - 1] : 0.);
    out[i] = static_cast<float>(sum);
  }
}

/// The vertical window sums from the transposed prefix of window_kernel_cols, as the root of the mean over the whole
/// box. Index i walks down the columns, so that the prefix is read coalesced and the strided accesses are the writes.
/// Returns each pixel's RMS as the median's first pass reads it.
struct WindowDiffCols
{
  const double *in;
  size_t n_y;
  size_t n_x;
  long box_size;
  float *out;

  __device__ float operator()(const size_t i) const
  {
    const auto x = i / n_y;
    const auto y = i % n_y;
    const long radius = box_size / 2;
    const long window_size = box_size > 0 ? box_size : 1;
    const long lo_y = min_of(y, radius);
    const long hi_y = max_of(y, window_size, radius, n_y);
    const long lo_x = min_of(x, radius);
    const long hi_x = max_of(x, window_size, radius, n_x);
    const auto column_offset = x * n_y;
    const double sum = in[column_offset + hi_y - 1] - (lo_y > 0 ? in[column_offset + lo_y - 1] : 0.);
    const long count = (hi_y - lo_y) * (hi_x - lo_x);
    const auto value = static_cast<float>(sqrt(sum / static_cast<double>(count)));
    out[y * n_x + x] = value;
    return value;
  }
  __device__ float4 load4(const size_t i4) const
  {
    return {(*this)(4 * i4), (*this)(4 * i4 + 1), (*this)(4 * i4 + 2), (*this)(4 * i4 + 3)};
  }
};

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
} // namespace

void gpu_rms_noise_map(const float *plane, size_t n_y, size_t n_x, long boxsize, float *row_window_sums,
                       double *row_prefix, DeviceVector<std::byte> &row_prefix_scan_storage, float *out_rms,
                       float *out_median, MedianScratch &scratch, cudaStream_t stream)
{
  assert(n_y > 0 && n_x > 0);

  // The horizontal pass: the per-row prefix sums of the squares, then every pixel's window sum from two of them.
  window_kernel(plane, n_y, n_x, row_prefix, row_prefix_scan_storage, stream);
  launch_per_element(window_diff, n_y * n_x, stream, row_prefix, n_y, n_x, boxsize, row_window_sums);
  // The vertical pass: the prefix sums down each column of those sums, then the root of the mean from two of them,
  // as the first pass of the RMS median.
  window_kernel_cols(row_window_sums, n_y, n_x, row_prefix, row_prefix_scan_storage, stream);
  radix_median(WindowDiffCols{row_prefix, n_y, n_x, boxsize, out_rms}, out_rms, n_y * n_x, false, 1.F, out_median,
               scratch, stream);
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