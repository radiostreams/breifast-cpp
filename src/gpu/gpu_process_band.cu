#include "gpu_band.hpp"

#include <cassert>
#include <tuple>
#include <utility>
#include <vector>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
namespace
{
int read_flag(int *flag, cudaStream_t stream)
{
  int host_flag = 0;
  CHECK_GPU(cudaMemcpyAsync(&host_flag, flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaStreamSynchronize(stream));
  return host_flag;
}

unsigned int read_count(const unsigned int *count, cudaStream_t stream)
{
  unsigned int host_count = 0;
  CHECK_GPU(cudaMemcpyAsync(&host_count, count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaStreamSynchronize(stream));
  return host_count;
}

/// Resizes `host` to `count` and copies the first `count` entries of `device` into it, on `stream`.
template <typename T>
void copy_column(const DeviceVector<T> &device, std::vector<T> &host, std::size_t count, cudaStream_t stream)
{
  assert(count <= device.size());
  host.resize(count);
  if (count > 0)
  {
    CHECK_GPU(cudaMemcpyAsync(host.data(), device.data(), count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }
}

template <typename Device, typename Host, std::size_t... I>
void copy_columns(const Device &device, const Host &host, std::size_t count, cudaStream_t stream,
                  std::index_sequence<I...>)
{
  (copy_column(std::get<I>(device), std::get<I>(host), count, stream), ...);
}

template <typename T> void copy_scalar(const DeviceVector<T> &device, T &host, cudaStream_t stream)
{
  CHECK_GPU(cudaMemcpyAsync(&host, device.data(), sizeof(T), cudaMemcpyDeviceToHost, stream));
}
} // namespace

bool gpu_process_band(const GpuPlaneInput &plane, const GpuBand &band, GpuBandWorkspace &workspace,
                      GpuBandResult &result, cudaStream_t stream)
{
  const std::size_t n_pixels = plane.n_pixels();
  assert(n_pixels > 0 && plane.n_channels > 0);
  assert(band.start >= 0 && band.start < band.stop && static_cast<std::size_t>(band.stop) <= plane.n_channels);
  assert(plane.max_sigma_bin > 0);
  assert(plane.boxsize >= 0 && plane.inner_boxsize >= 0);
  assert(plane.cube != nullptr && plane.weights != nullptr && plane.accept_mask != nullptr &&
         plane.selfnoise_offset != nullptr && plane.selfnoise_raw != nullptr && plane.dde_noise != nullptr);
  assert(band.beam_kernel == nullptr || (band.beam_halfsize >= 0 && plane.max_to_std_ratio > 0.F));

  const float inv_band_size = 1.F / static_cast<float>(band.stop - band.start);
  const auto histogram_bins = static_cast<std::size_t>(2 * plane.max_sigma_bin);
  workspace.ensure(plane);
  result.ensure(0, histogram_bins); // the island columns grow once the islands are counted

  {
    const NvtxRange range{"mask"};
    gpu_mask_or(plane.accept_mask, n_pixels, band.start, band.stop, workspace.mask, workspace.any_flag, stream);
    if (read_flag(workspace.any_flag, stream) == 0)
    {
      return false;
    }
  }

  {
    const NvtxRange range{"band image and noise"};
    gpu_weighted_band_image(plane.cube, plane.weights, plane.mean_image, workspace.plane, n_pixels, band.start,
                            band.stop, inv_band_size, stream);

    // Both noise modes need it: the band result reports it, and it is the base noise without an RMS map.
    gpu_median_abs(workspace.plane, n_pixels, result.global_mad, 1.4826F, workspace.median_scratch, stream);
    const float *rms = nullptr;
    const float *base_noise = result.global_mad;
    if (plane.use_rms_noise_estimator)
    {
      gpu_rms_noise_map(workspace.plane, plane.n_y, plane.n_x, plane.boxsize, workspace.row_sums, workspace.rms_map,
                        stream);
      gpu_median(workspace.rms_map, n_pixels, result.rms_floor, workspace.median_scratch, stream);
      rms = workspace.rms_map;
      base_noise = result.rms_floor;
    }
    else
    {
      CHECK_GPU(cudaMemsetAsync(result.rms_floor, 0, sizeof(float), stream)); // 0 in the global MAD mode, as on the CPU
    }
    gpu_combine_noise_map(rms, base_noise, plane.selfnoise_offset, plane.dde_noise, band.selfnoise_scale_factor,
                          plane.ddenoise_scaling_factor, workspace.noise_map, n_pixels, stream);

    gpu_median(workspace.noise_map, n_pixels, result.median_noise, workspace.median_scratch, stream);
  }

  {
    const NvtxRange range{"snr"};
    common::SnrFilterParams params;
    params.inv_band_size = inv_band_size;
    params.band_start = band.start;
    params.band_stop = band.stop;
    params.max_sigma_bin = plane.max_sigma_bin;
    params.histogram_detail = plane.histogram_detail;
    params.reject_noise_threshold = plane.reject_noise_threshold;
    params.threshold = plane.threshold;

    gpu_snr_and_filters(workspace.plane, workspace.noise_map, plane.snr_cutoff, plane.flux_cutoff, result.median_noise,
                        n_pixels, params, workspace.snr, workspace.mask, result.histogram0, result.histogram1,
                        result.histogram2, result.min_sigma, result.max_sigma, stream);

    // As on the CPU: a single-channel band of a multi-channel plane writes its SNR into its channel of the cube, which
    // the bands after it and the spectral variance stats then read.
    if (band.stop - band.start == 1 && plane.n_channels > 1)
    {
      CHECK_GPU(cudaMemcpyAsync(plane.cube + band.start * n_pixels, workspace.snr, n_pixels * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
    }
  }

  unsigned int n_islands = 0;
  {
    const NvtxRange range{"label"};
    // An empty mask labels no island, so the count also tells whether anything passed the filters.
    gpu_label_islands(workspace.mask, plane.n_y, plane.n_x, workspace.labels, workspace.label_count,
                      workspace.label_scratch, stream);
    n_islands = read_count(workspace.label_count, stream);
  }
  result.label_count = n_islands;
  if (n_islands == 0)
  {
    return true;
  }
  workspace.island_scratch.ensure(n_islands);
  result.ensure(n_islands, histogram_bins);

  {
    const NvtxRange range{"island stats"};
    gpu_label_stats(workspace.labels, workspace.snr, workspace.noise_map, n_pixels, plane.n_x, n_islands,
                    result.islands.peak_value, result.islands.peak_local_noise, result.islands.peak_y,
                    result.islands.peak_x, result.islands.island_size, workspace.island_scratch, stream);

    // The donut around each peak, with the inner box zeroed, and the beam window, kept whole: both read in place.
    const StampWindows donut{
        .plane = workspace.plane,
        .n_y = plane.n_y,
        .n_x = plane.n_x,
        .peak_y = result.islands.peak_y,
        .peak_x = result.islands.peak_x,
        .radius = plane.boxsize / 2,
        .inner_half = plane.inner_boxsize,
    };
    gpu_stamp_abs_extreme(donut, n_islands, result.islands.stamp_extreme, stream);
    gpu_donut_std(donut, n_islands, result.islands.donut_std, stream);
    gpu_donut_median_abs(donut, n_islands, 1.4826F, result.islands.donut_lmad, stream);

    if (band.beam_kernel != nullptr)
    {
      StampWindows beam = donut;
      beam.radius = band.beam_halfsize;
      beam.inner_half = -1;
      gpu_stamp_beam_chi2(beam, band.beam_kernel, band.beam_kernel_sum, n_islands, result.islands.stamp_extreme,
                          1.F / plane.max_to_std_ratio, result.islands.beam_chi2, stream);
    }
    else
    {
      CHECK_GPU(cudaMemsetAsync(result.islands.beam_chi2, 0, n_islands * sizeof(float), stream));
    }

    gpu_gather_at_peaks(
        workspace.plane, workspace.noise_map, plane.use_rms_noise_estimator ? workspace.rms_map.data() : nullptr,
        plane.selfnoise_raw, plane.n_x, result.islands.peak_y, result.islands.peak_x, n_islands, result.islands.flux,
        result.islands.noise_at_peak, result.islands.rms_at_peak, result.islands.selfnoise_at_peak, stream);

    if (plane.baseline_flux != nullptr)
    {
      gpu_island_boost(workspace.labels, plane.baseline_flux, n_pixels, n_islands, result.islands.flux,
                       result.islands.boost, workspace.island_scratch, stream);
    }
    else
    {
      CHECK_GPU(cudaMemsetAsync(result.islands.boost, 0, n_islands * sizeof(float), stream));
    }
  }

  return true;
}

void GpuBandWorkspace::ensure(const GpuPlaneInput &input)
{
  const std::size_t n_pixels = input.n_pixels();
  plane.ensure(n_pixels);
  noise_map.ensure(n_pixels);
  snr.ensure(n_pixels);
  mask.ensure(n_pixels);
  row_sums.ensure(n_pixels);
  rms_map.ensure(n_pixels);

  labels.ensure(n_pixels);
  label_scratch.ensure(n_pixels);
}

void GpuBandResult::ensure(const std::size_t n_islands, const std::size_t histogram_bins)
{
  std::apply([n_islands](auto &...column) { (column.ensure(n_islands), ...); }, islands.tie());
  histogram0.ensure(histogram_bins);
  histogram1.ensure(histogram_bins);
  histogram2.ensure(histogram_bins);
  this->histogram_bins = histogram_bins;
}

void copy_band_result(const GpuBandResult &source, GpuBandHostResult &destination, cudaStream_t stream)
{
  const NvtxRange range{"copy result"};
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
