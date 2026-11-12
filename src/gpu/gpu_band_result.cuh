#pragma once

#include <cstddef>
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

/// Copies the band outputs of `source` into `destination` and waits for `stream`.
void copy_band_result(const GpuBandResult &source, GpuBandHostResult &destination, GpuStream stream);
} // namespace breifast::gpu
