#include "gpu_band_result.cuh"

#include <algorithm>
#include <cassert>
#include <tuple>
#include <utility>
#include <vector>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
namespace
{
/// Resizes `host` to `count` and copies the first `count` entries of `device` into it, on `stream`.
template <typename T>
void copy_column(const DeviceVector<T> &device, std::vector<T> &host, std::size_t count, GpuStream stream)
{
  assert(count <= device.size());
  host.resize(count);
  if (count > 0)
  {
    CHECK_GPU(cudaMemcpyAsync(host.data(), device.data(), count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }
}

template <typename Device, typename Host, std::size_t... I>
void copy_columns(const Device &device, const Host &host, std::size_t count, GpuStream stream,
                  std::index_sequence<I...>)
{
  (copy_column(std::get<I>(device), std::get<I>(host), count, stream), ...);
}

template <typename T> void copy_scalar(const DeviceVector<T> &device, T &host, GpuStream stream)
{
  CHECK_GPU(cudaMemcpyAsync(&host, device.data(), sizeof(T), cudaMemcpyDeviceToHost, stream));
}
} // namespace

void GpuBandWorkspace::ensure(const GpuPlaneInput &input, const GpuBand &band)
{
  const std::size_t n_pixels = input.n_pixels();
  const std::size_t island_capacity = std::min(n_pixels, max_stamp_capacity);
  const auto stamp_side = static_cast<std::size_t>(2 * (input.boxsize / 2) + 1);

  plane.ensure(n_pixels);
  noise_map.ensure(n_pixels);
  snr.ensure(n_pixels);
  mask.ensure(n_pixels);
  row_sums.ensure(n_pixels);
  rms_map.ensure(n_pixels);

  labels.ensure(n_pixels);
  label_scratch.ensure(n_pixels);
  island_scratch.ensure(island_capacity);

  donut_stamps.ensure(island_capacity * stamp_side * stamp_side);
  if (band.beam_kernel != nullptr)
  {
    const auto beam_side = static_cast<std::size_t>(2 * band.beam_halfsize + 1);
    beam_stamps.ensure(island_capacity * beam_side * beam_side);
  }
}

void GpuBandResult::ensure(const std::size_t island_capacity, const std::size_t histogram_bins)
{
  std::apply([island_capacity](auto &...column) { (column.ensure(island_capacity), ...); }, islands.tie());
  histogram0.ensure(histogram_bins);
  histogram1.ensure(histogram_bins);
  histogram2.ensure(histogram_bins);
  this->histogram_bins = histogram_bins;
}

void copy_band_result(const GpuBandResult &source, GpuBandHostResult &destination, GpuStream stream)
{
  const std::size_t count = source.label_count;
  destination.label_count = count;
  const auto host_columns = destination.islands.tie();
  copy_columns(source.islands.tie(), host_columns, count, stream,
               std::make_index_sequence<std::tuple_size_v<decltype(host_columns)>>{});

  copy_column(source.histogram0, destination.histogram0, source.histogram_bins, stream);
  copy_column(source.histogram1, destination.histogram1, source.histogram_bins, stream);
  copy_column(source.histogram2, destination.histogram2, source.histogram_bins, stream);

  copy_scalar(source.min_sigma, destination.min_sigma, stream);
  copy_scalar(source.max_sigma, destination.max_sigma, stream);
  copy_scalar(source.median_noise, destination.median_noise, stream);
  copy_scalar(source.rms_floor, destination.rms_floor, stream);
  copy_scalar(source.global_mad, destination.global_mad, stream);

  CHECK_GPU(cudaStreamSynchronize(stream));
}
} // namespace breifast::gpu
