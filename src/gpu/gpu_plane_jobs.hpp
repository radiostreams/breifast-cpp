#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "device_vector.hpp"
#include "gpu_band_workspace.cuh"
#include "gpu_plane_detections.hpp"

namespace breifast::gpu
{
struct GpuBandSpec
{
  long start = 0;
  long stop = 0;
  float selfnoise_scale_factor = 1.F;
  std::string label;
};

struct GpuPlaneJobSpec
{
  const float *cube = nullptr;
  const float *weights = nullptr;
  const bool *nonzero_channels = nullptr;
  const bool *accept_mask = nullptr;
  const float *mean_image = nullptr;
  const float *selfnoise_offset = nullptr;
  const float *selfnoise_raw = nullptr;
  const float *dde_noise = nullptr;
  const float *snr_cutoff = nullptr;
  const float *flux_cutoff = nullptr;
  const float *baseline_flux = nullptr;
  const float *beam_major_axis = nullptr;
  const float *beam_minor_axis = nullptr;
  const float *beam_position_angle = nullptr;
  const float *fixed_beam_kernel = nullptr;

  std::string stokes_label;
  std::vector<GpuBandSpec> bands;

  std::size_t n_y = 0;
  std::size_t n_x = 0;
  std::size_t n_channels = 0;

  long fixed_beam_halfsize = 0;
  long boxsize = 0;
  long inner_boxsize = 0;
  long histogram_detail = 0;
  long max_sigma_bin = 200;

  float ddenoise_scaling_factor = 1.F;
  float reject_noise_threshold = 0.F;
  float threshold = 0.F;
  float max_to_std_ratio_value = 1.F;

  bool use_rms_noise_estimator = true;
  bool subtract_mean = false;
  bool beam_size_enabled = false;
};

struct GpuPlaneSizes
{
  std::size_t n_pixels = 0;
  std::size_t channel_pixels = 0;
};

std::vector<float> build_gaussian_beam_kernel_vector(long halfsize, float sigma_major, float sigma_minor,
                                                     float rotate);

/// Sum of a (2 * halfsize + 1)^2 beam kernel, with the same xtensor expression as the CPU stamp_beam_chi2, so that both
/// backends divide by the same float.
float beam_kernel_sum(const float *kernel, long halfsize);

struct GpuPlaneJobs
{
  explicit GpuPlaneJobs(const GpuPlaneJobSpec &spec);

  GpuPlaneJobs(const GpuPlaneJobs &) = delete;
  GpuPlaneJobs &operator=(const GpuPlaneJobs &) = delete;
  GpuPlaneJobs(GpuPlaneJobs &&) noexcept = default;
  GpuPlaneJobs &operator=(GpuPlaneJobs &&) noexcept = default;
  ~GpuPlaneJobs() = default;

  std::size_t n_y = 0;
  std::size_t n_x = 0;
  std::size_t n_channels = 0;
  float max_to_std_ratio_value = 1.F;
  bool use_rms_noise_estimator = true;

  DeviceVector<float> cube;
  DeviceVector<float> weights;
  DeviceVector<bool> nonzero_channels;
  DeviceVector<bool> accept_mask;
  DeviceVector<float> selfnoise_offset;
  DeviceVector<float> selfnoise_raw;
  DeviceVector<float> dde_noise;
  std::optional<DeviceVector<float>> mean_image;
  std::optional<DeviceVector<float>> snr_cutoff;
  std::optional<DeviceVector<float>> flux_cutoff;
  std::optional<DeviceVector<float>> baseline_flux;
  std::optional<DeviceVector<float>> fixed_beam_kernel;
  std::vector<DeviceVector<float>> band_beam_kernels;

  // One entry per band, in config.band_slices order.
  std::vector<GpuBandInput> band_inputs;
  std::vector<GpuBandWorkspaceSizes> band_workspace_sizes;
  std::vector<GpuBandLabels> labels;
  std::size_t histogram_bins = 0;

private:
  GpuPlaneJobs(const GpuPlaneJobSpec &spec, const GpuPlaneSizes &sizes);
};
} // namespace breifast::gpu
