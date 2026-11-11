#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_vector.hpp"
#include "gpu_kernels.hpp"

namespace breifast::gpu
{
inline constexpr std::size_t max_stamp_capacity = 4096;

struct GpuBandInputSizes
{
  std::size_t cube = 0;
  std::size_t weights = 0;
  std::size_t nonzero_channels = 0;
  std::size_t accept_mask = 0;
  std::size_t mean_image = 0;
  std::size_t selfnoise_offset = 0;
  std::size_t selfnoise_raw = 0;
  std::size_t dde_noise = 0;
  std::size_t snr_cutoff = 0;
  std::size_t flux_cutoff = 0;
  std::size_t baseline_flux = 0;
  std::size_t beam_kernel = 0;
};

struct GpuBandInput
{
  GpuBandInputSizes sizes{};
  const float *cube = nullptr;
  const float *weights = nullptr;
  const bool *nonzero_channels = nullptr;
  const float *mean_image = nullptr;
  const bool *accept_mask = nullptr;
  const float *selfnoise_offset = nullptr;
  const float *selfnoise_raw = nullptr;
  const float *dde_noise = nullptr;
  const float *snr_cutoff = nullptr;
  const float *flux_cutoff = nullptr;
  const float *baseline_flux = nullptr;
  const float *beam_kernel = nullptr;

  std::size_t n_y = 0;
  std::size_t n_x = 0;
  std::size_t n_pixels = 0;
  std::size_t n_channels = 0;

  long band_start = 0;
  long band_stop = 0;
  long boxsize = 0;
  long inner_boxsize = 0;
  long beam_halfsize = 0;
  long histogram_detail = 0;
  long max_sigma_bin = 200;

  float selfnoise_scale_factor = 1.F;
  float inv_band_size = 1.F;
  float ddenoise_scaling_factor = 1.F;
  float reject_noise_threshold = 0.F;
  float threshold = 0.F;
  float max_to_std_ratio_value = 0.F;
  float beam_kernel_sum = 0.F;

  bool use_rms_noise_estimator = false;
  bool subtract_mean = false;
};

struct GpuBandWorkspaceSizes
{
  std::size_t n_pixels = 0;
  std::size_t stamp_capacity = 0;
  std::size_t stamp_pixels = 0;
  std::size_t beam_stamp_pixels = 0;
  std::size_t histogram_bins = 0;
};

inline void validate_band_workspace_sizes(const GpuBandWorkspaceSizes &sizes)
{
  if (sizes.n_pixels == 0)
  {
    throw std::runtime_error("GpuBandWorkspace: n_pixels must be greater than 0");
  }
  if (sizes.stamp_pixels == 0)
  {
    throw std::runtime_error("GpuBandWorkspace: stamp_pixels must be greater than 0");
  }
  if (sizes.stamp_capacity == 0)
  {
    throw std::runtime_error("GpuBandWorkspace: stamp_capacity must be greater than 0");
  }
  if (sizes.stamp_capacity > max_stamp_capacity)
  {
    throw std::runtime_error("GpuBandWorkspace: stamp_capacity exceeds the maximum of " +
                             std::to_string(max_stamp_capacity));
  }
}

/// Device scratch of gpu_median and gpu_median_abs: the 256-bin digit histogram, then the radix-select state (count
/// below the prefix, prefix, coverage) and the largest key below the median.
struct MedianScratch
{
  static constexpr std::size_t histogram_bins = 256;
  DeviceVector<std::uint32_t> words{histogram_bins + 4};
};

struct GpuBandWorkspace
{
  explicit GpuBandWorkspace(const GpuBandWorkspaceSizes &sizes);

  GpuBandWorkspace(const GpuBandWorkspace &) = delete;
  GpuBandWorkspace &operator=(const GpuBandWorkspace &) = delete;
  GpuBandWorkspace(GpuBandWorkspace &&) noexcept;
  GpuBandWorkspace &operator=(GpuBandWorkspace &&) noexcept;
  ~GpuBandWorkspace();

  GpuBandWorkspaceSizes sizes;

  DeviceVector<float> plane;
  DeviceVector<float> noise_map;
  DeviceVector<float> snr;
  DeviceVector<bool> mask;
  DeviceVector<float> row_sums;
  DeviceVector<float> rms_map;

  DeviceVector<unsigned int> labels;
  DeviceVector<unsigned int> label_scratch;

  DeviceVector<float> donut_stamps;
  DeviceVector<float> beam_stamps;

  DeviceVector<int> any_flag;
  DeviceVector<unsigned int> label_count;

  MedianScratch median_scratch;
};

struct GpuBandResult
{
  GpuBandResult(std::size_t n_pixels, std::size_t histogram_bins);

  GpuBandResult(const GpuBandResult &) = delete;
  GpuBandResult &operator=(const GpuBandResult &) = delete;
  GpuBandResult(GpuBandResult &&) noexcept;
  GpuBandResult &operator=(GpuBandResult &&) noexcept;
  ~GpuBandResult();

  std::size_t n_pixels = 0;
  std::size_t histogram_bins = 0;
  unsigned int label_count = 0;

  DeviceVector<float> peak_value;
  DeviceVector<float> peak_local_noise;
  DeviceVector<long> peak_y;
  DeviceVector<long> peak_x;
  DeviceVector<unsigned int> island_size;

  DeviceVector<float> stamp_extreme;
  DeviceVector<float> donut_std;
  DeviceVector<float> donut_lmad;
  DeviceVector<float> beam_chi2;
  DeviceVector<float> flux;
  DeviceVector<float> boost;
  DeviceVector<bool> boost_found;

  DeviceVector<float> noise_at_peak;
  DeviceVector<float> rms_at_peak;
  DeviceVector<float> selfnoise_at_peak;

  DeviceVector<unsigned long long> histogram0;
  DeviceVector<unsigned long long> histogram1;
  DeviceVector<unsigned long long> histogram2;

  DeviceVector<float> min_sigma;
  DeviceVector<float> max_sigma;
  DeviceVector<float> median_noise;
  DeviceVector<float> rms_floor;
  DeviceVector<float> global_mad;
};
} // namespace breifast::gpu
