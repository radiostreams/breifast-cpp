#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "gpu_pipeline.hpp"

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
  GpuBandInputSizes sizes;
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
  float inv_weights_sum = 1.F;
  float ddenoise_scaling_factor = 1.F;
  float reject_noise_threshold = 0.F;
  float threshold = 0.F;
  float max_to_std_ratio_value = 0.F;

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

struct GpuBandWorkspace
{
  GpuBandWorkspaceSizes sizes;

  float *plane = nullptr;
  float *noise_map = nullptr;
  float *snr = nullptr;
  bool *mask = nullptr;
  float *row_sums = nullptr;
  float *rms_map = nullptr;

  unsigned int *labels = nullptr;
  unsigned int *label_scratch = nullptr;

  // stamp_capacity * stamp_pixels
  float *donut_stamps = nullptr;
  // stamp_capacity * beam_stamp_pixels
  float *beam_stamps = nullptr;

  int *any_flag = nullptr;
  unsigned int *label_count = nullptr;
};

struct GpuBandResult
{
  std::size_t n_pixels = 0;
  std::size_t histogram_bins = 0;
  unsigned int label_count = 0;

  float *peak_value = nullptr;
  float *peak_local_noise = nullptr;
  long *peak_y = nullptr;
  long *peak_x = nullptr;
  unsigned int *island_size = nullptr;

  float *stamp_extreme = nullptr;
  float *donut_std = nullptr;
  float *donut_lmad = nullptr;
  float *beam_chi2 = nullptr;
  float *flux = nullptr;
  float *boost = nullptr;
  bool *boost_found = nullptr;

  float *noise_at_peak = nullptr;
  float *rms_at_peak = nullptr;
  float *selfnoise_at_peak = nullptr;

  unsigned long long *histogram0 = nullptr;
  unsigned long long *histogram1 = nullptr;
  unsigned long long *histogram2 = nullptr;

  float *min_sigma = nullptr;
  float *max_sigma = nullptr;
  float *median_noise = nullptr;
  float *rms_floor = nullptr;
  float *global_mad = nullptr;
};

GpuBandWorkspace create_band_workspace(const GpuBandWorkspaceSizes &sizes, GpuStream stream);
void destroy_band_workspace(GpuBandWorkspace &workspace, GpuStream stream);

GpuBandResult create_band_result(std::size_t n_pixels, std::size_t histogram_bins, GpuStream stream);
void destroy_band_result(GpuBandResult &result, GpuStream stream);
} // namespace breifast::gpu
