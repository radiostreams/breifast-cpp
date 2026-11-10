#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "gpu_band_workspace.cuh"
#include "gpu_process_band.cuh"

namespace breifast::gpu
{
struct GpuBandHostResult
{
  std::size_t label_count = 0;

  std::vector<float> peak_value;
  std::vector<float> peak_local_noise;
  std::vector<long> peak_y;
  std::vector<long> peak_x;
  std::vector<unsigned int> island_size;

  std::vector<float> stamp_extreme;
  std::vector<float> donut_std;
  std::vector<float> donut_lmad;
  std::vector<float> beam_chi2;
  std::vector<float> flux;
  std::vector<float> boost;
  std::vector<unsigned char> boost_found;

  std::vector<float> noise_at_peak;
  std::vector<float> rms_at_peak;
  std::vector<float> selfnoise_at_peak;

  std::vector<unsigned long long> histogram0;
  std::vector<unsigned long long> histogram1;
  std::vector<unsigned long long> histogram2;

  float min_sigma = 0.F;
  float max_sigma = 0.F;
  float median_noise = 0.F;
  float rms_floor = 0.F;
  float global_mad = 0.F;
};

void copy_band_result(const GpuBandResult &source, GpuBandHostResult &destination, GpuStream stream);

void copy_band_jobs(std::vector<GpuBandJob> &jobs, std::vector<GpuBandHostResult> &results);

struct GpuPlaneHostResult
{
  std::vector<GpuBandStatus> statuses;
  std::vector<GpuBandHostResult> bands;

  std::array<std::vector<unsigned long long>, 3> histograms;
  std::vector<double> median_noises;
  std::optional<float> min_sigma;
  std::optional<float> max_sigma;
};

void assemble_plane_bands(const std::vector<GpuBandStatus> &statuses,
                          const std::vector<GpuBandHostResult> &bands, GpuPlaneHostResult &result);

void gpu_process_plane(std::vector<GpuBandJob> &jobs, GpuPlaneHostResult &result);
} // namespace breifast::gpu
