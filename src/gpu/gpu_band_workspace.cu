#include "gpu_band_workspace.cuh"

namespace breifast::gpu
{
GpuBandWorkspace::GpuBandWorkspace(const GpuBandWorkspaceSizes &sizes) : sizes(sizes)
{
  validate_band_workspace_sizes(this->sizes);

  plane.resize(this->sizes.n_pixels);
  noise_map.resize(this->sizes.n_pixels);
  snr.resize(this->sizes.n_pixels);
  mask.resize(this->sizes.n_pixels);
  row_sums.resize(this->sizes.n_pixels);
  rms_map.resize(this->sizes.n_pixels);

  labels.resize(this->sizes.n_pixels);
  label_scratch.resize(this->sizes.n_pixels);

  donut_stamps.resize(this->sizes.stamp_capacity * this->sizes.stamp_pixels);
  if (this->sizes.beam_stamp_pixels > 0)
  {
    beam_stamps.resize(this->sizes.stamp_capacity * this->sizes.beam_stamp_pixels);
  }

  any_flag.resize(1);
  label_count.resize(1);
}

GpuBandWorkspace::GpuBandWorkspace(GpuBandWorkspace &&) noexcept = default;
GpuBandWorkspace &GpuBandWorkspace::operator=(GpuBandWorkspace &&) noexcept = default;
GpuBandWorkspace::~GpuBandWorkspace() = default;

GpuBandResult::GpuBandResult(std::size_t n_pixels, std::size_t histogram_bins)
    : n_pixels(n_pixels), histogram_bins(histogram_bins)
{
  if (n_pixels == 0)
  {
    throw std::runtime_error("GpuBandResult: n_pixels must be greater than 0");
  }
  if (histogram_bins == 0)
  {
    throw std::runtime_error("GpuBandResult: histogram_bins must be greater than 0");
  }

  peak_value.resize(n_pixels);
  peak_local_noise.resize(n_pixels);
  peak_y.resize(n_pixels);
  peak_x.resize(n_pixels);
  island_size.resize(n_pixels);

  stamp_extreme.resize(n_pixels);
  donut_std.resize(n_pixels);
  donut_lmad.resize(n_pixels);
  beam_chi2.resize(n_pixels);
  flux.resize(n_pixels);
  boost.resize(n_pixels);
  boost_found.resize(n_pixels);

  noise_at_peak.resize(n_pixels);
  rms_at_peak.resize(n_pixels);
  selfnoise_at_peak.resize(n_pixels);

  histogram0.resize(histogram_bins);
  histogram1.resize(histogram_bins);
  histogram2.resize(histogram_bins);

  min_sigma.resize(1);
  max_sigma.resize(1);
  median_noise.resize(1);
  rms_floor.resize(1);
  global_mad.resize(1);
}

GpuBandResult::GpuBandResult(GpuBandResult &&) noexcept = default;
GpuBandResult &GpuBandResult::operator=(GpuBandResult &&) noexcept = default;
GpuBandResult::~GpuBandResult() = default;
} // namespace breifast::gpu
