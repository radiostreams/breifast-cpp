#include "gpu_band_result.cuh"

#include <cassert>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
GpuBandWorkspace::GpuBandWorkspace(const GpuBandWorkspaceSizes &sizes) : sizes(sizes)
{
  assert(this->sizes.n_pixels > 0);
  assert(this->sizes.stamp_pixels > 0);
  assert(this->sizes.stamp_capacity > 0 && this->sizes.stamp_capacity <= max_stamp_capacity);

  plane.ensure(this->sizes.n_pixels);
  noise_map.ensure(this->sizes.n_pixels);
  snr.ensure(this->sizes.n_pixels);
  mask.ensure(this->sizes.n_pixels);
  row_sums.ensure(this->sizes.n_pixels);
  rms_map.ensure(this->sizes.n_pixels);

  labels.ensure(this->sizes.n_pixels);
  label_scratch = LabelScratch(this->sizes.n_pixels);
  island_scratch.ensure(this->sizes.stamp_capacity);

  donut_stamps.ensure(this->sizes.stamp_capacity * this->sizes.stamp_pixels);
  if (this->sizes.beam_stamp_pixels > 0)
  {
    beam_stamps.ensure(this->sizes.stamp_capacity * this->sizes.beam_stamp_pixels);
  }

  any_flag.ensure(1);
  label_count.ensure(1);
}

GpuBandWorkspace::GpuBandWorkspace(GpuBandWorkspace &&) noexcept = default;
GpuBandWorkspace &GpuBandWorkspace::operator=(GpuBandWorkspace &&) noexcept = default;
GpuBandWorkspace::~GpuBandWorkspace() = default;

GpuBandResult::GpuBandResult(std::size_t n_pixels, std::size_t histogram_bins)
    : n_pixels(n_pixels), histogram_bins(histogram_bins)
{
  assert(n_pixels > 0);
  assert(histogram_bins > 0);

  peak_value.ensure(n_pixels);
  peak_local_noise.ensure(n_pixels);
  peak_y.ensure(n_pixels);
  peak_x.ensure(n_pixels);
  island_size.ensure(n_pixels);

  stamp_extreme.ensure(n_pixels);
  donut_std.ensure(n_pixels);
  donut_lmad.ensure(n_pixels);
  beam_chi2.ensure(n_pixels);
  flux.ensure(n_pixels);
  boost.ensure(n_pixels);
  boost_found.ensure(n_pixels);

  noise_at_peak.ensure(n_pixels);
  rms_at_peak.ensure(n_pixels);
  selfnoise_at_peak.ensure(n_pixels);

  histogram0.ensure(histogram_bins);
  histogram1.ensure(histogram_bins);
  histogram2.ensure(histogram_bins);

  min_sigma.ensure(1);
  max_sigma.ensure(1);
  median_noise.ensure(1);
  rms_floor.ensure(1);
  global_mad.ensure(1);
}

GpuBandResult::GpuBandResult(GpuBandResult &&) noexcept = default;
GpuBandResult &GpuBandResult::operator=(GpuBandResult &&) noexcept = default;
GpuBandResult::~GpuBandResult() = default;

void copy_band_result(const GpuBandResult &source, GpuBandHostResult &destination, GpuStream stream)
{
  assert(source.label_count <= source.n_pixels);
  assert(source.histogram_bins > 0);

  destination.label_count = source.label_count;
  const std::size_t count = source.label_count;
  const std::size_t bins = source.histogram_bins;

  destination.peak_value.resize(count);
  destination.peak_local_noise.resize(count);
  destination.peak_y.resize(count);
  destination.peak_x.resize(count);
  destination.island_size.resize(count);
  destination.stamp_extreme.resize(count);
  destination.donut_std.resize(count);
  destination.donut_lmad.resize(count);
  destination.beam_chi2.resize(count);
  destination.flux.resize(count);
  destination.boost.resize(count);
  destination.boost_found.resize(count);
  destination.noise_at_peak.resize(count);
  destination.rms_at_peak.resize(count);
  destination.selfnoise_at_peak.resize(count);

  destination.histogram0.resize(bins);
  destination.histogram1.resize(bins);
  destination.histogram2.resize(bins);

  if (count > 0)
  {
    CHECK_GPU(cudaMemcpyAsync(destination.peak_value.data(), source.peak_value, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.peak_local_noise.data(), source.peak_local_noise, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.peak_y.data(), source.peak_y, count * sizeof(long), cudaMemcpyDeviceToHost,
                              stream));
    CHECK_GPU(cudaMemcpyAsync(destination.peak_x.data(), source.peak_x, count * sizeof(long), cudaMemcpyDeviceToHost,
                              stream));
    CHECK_GPU(cudaMemcpyAsync(destination.island_size.data(), source.island_size, count * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.stamp_extreme.data(), source.stamp_extreme, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.donut_std.data(), source.donut_std, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.donut_lmad.data(), source.donut_lmad, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.beam_chi2.data(), source.beam_chi2, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(
        cudaMemcpyAsync(destination.flux.data(), source.flux, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(
        cudaMemcpyAsync(destination.boost.data(), source.boost, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.boost_found.data(), source.boost_found, count * sizeof(bool),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.noise_at_peak.data(), source.noise_at_peak, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.rms_at_peak.data(), source.rms_at_peak, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CHECK_GPU(cudaMemcpyAsync(destination.selfnoise_at_peak.data(), source.selfnoise_at_peak, count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
  }

  CHECK_GPU(cudaMemcpyAsync(destination.histogram0.data(), source.histogram0, bins * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaMemcpyAsync(destination.histogram1.data(), source.histogram1, bins * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaMemcpyAsync(destination.histogram2.data(), source.histogram2, bins * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, stream));

  CHECK_GPU(cudaMemcpyAsync(&destination.min_sigma, source.min_sigma, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaMemcpyAsync(&destination.max_sigma, source.max_sigma, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(
      cudaMemcpyAsync(&destination.median_noise, source.median_noise, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaMemcpyAsync(&destination.rms_floor, source.rms_floor, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK_GPU(cudaMemcpyAsync(&destination.global_mad, source.global_mad, sizeof(float), cudaMemcpyDeviceToHost, stream));

  CHECK_GPU(cudaStreamSynchronize(stream));
}
} // namespace breifast::gpu
