#pragma once

#include <cstddef>
#include <string>
#include <vector>

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

struct GpuPlaneJobs
{
  std::vector<GpuBandJob> jobs;
  std::vector<GpuBandLabels> labels;
  std::vector<void *> buffers;

  const float *cube = nullptr;
  const bool *nonzero_channels = nullptr;
  std::size_t n_y = 0;
  std::size_t n_x = 0;
  std::size_t n_channels = 0;
  float max_to_std_ratio_value = 1.F;
  bool use_rms_noise_estimator = true;
};

GpuPlaneJobs build_plane_jobs(const GpuPlaneJobSpec &spec);
void destroy_plane_jobs(GpuPlaneJobs &plane_jobs);

void *gpu_upload_buffer(const void *host, std::size_t bytes);
void gpu_free_buffer(void *device) noexcept;
} // namespace breifast::gpu
