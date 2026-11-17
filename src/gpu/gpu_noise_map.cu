#include <cassert>
#include <climits>

#include <cub/device/device_scan.cuh>
#include <cuda/std/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "gpu_kernels.hpp"

#include "gpu_device_utils.cuh"

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
/// box. One thread per pixel, consecutive threads walking down a column, so that the prefix is read coalesced and the
/// strided accesses are the writes.
__global__ void window_diff_cols(const double *in, const size_t n_y, const size_t n_x, long box_size, float *out)
{
  const auto i = global_index();
  if (i < n_x * n_y)
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
    out[y * n_x + x] = static_cast<float>(sqrt(sum / static_cast<double>(count)));
  }
}

// The base noise is max(rms[p], *base_noise) with an RMS map, else *base_noise.
__global__ void combine_noise_map_kernel(const float *rms, const float *base_noise, const float *selfnoise_offset,
                                         const float *dde_noise, const float selfnoise_scale, const float dde_scale,
                                         float *out_noise_map, const size_t n_pixels)
{
  const auto i_pixel = global_index();
  if (i_pixel < n_pixels)
  {
    const float base = rms != nullptr ? fmaxf(rms[i_pixel], *base_noise) : *base_noise;
    out_noise_map[i_pixel] =
        sqrtf(base * base + selfnoise_scale * selfnoise_offset[i_pixel] + dde_scale * dde_noise[i_pixel]);
  }
}
} // namespace

void gpu_rms_noise_map(const float *plane, size_t n_y, size_t n_x, long boxsize, float *row_window_sums,
                       double *row_prefix, DeviceVector<std::byte> &row_prefix_scan_storage, float *out_rms,
                       cudaStream_t stream)
{
  assert(n_y > 0 && n_x > 0);

  // The horizontal pass: the per-row prefix sums of the squares, then every pixel's window sum from two of them.
  window_kernel(plane, n_y, n_x, row_prefix, row_prefix_scan_storage, stream);
  launch_per_element(window_diff, n_y * n_x, stream, row_prefix, n_y, n_x, boxsize, row_window_sums);
  // The vertical pass: the prefix sums down each column of those sums, then the root of the mean from two of them.
  window_kernel_cols(row_window_sums, n_y, n_x, row_prefix, row_prefix_scan_storage, stream);
  launch_per_element(window_diff_cols, n_y * n_x, stream, row_prefix, n_y, n_x, boxsize, out_rms);
}

void gpu_combine_noise_map(const float *rms, const float *base_noise, const float *selfnoise_offset,
                           const float *dde_noise, const float selfnoise_scale, const float dde_scale,
                           float *out_noise_map, const size_t n_pixels, cudaStream_t stream)
{
  assert(n_pixels > 0);
  launch_per_element(combine_noise_map_kernel, n_pixels, stream, rms, base_noise, selfnoise_offset, dde_noise,
                     selfnoise_scale, dde_scale, out_noise_map, n_pixels);
}
} // namespace breifast::gpu