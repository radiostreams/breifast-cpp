#pragma once

#include <cstddef>
#include <vector>

#include "gpu_band_workspace.cuh"
#include "gpu_process_band.cuh"

namespace breifast::gpu
{
/// Host copy of a GpuBandResult: every island column holds label_count entries.
struct GpuBandHostResult
{
  std::size_t label_count = 0;

  IslandColumns<std::vector> islands;

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

/// Device copies of a plane's inputs, grown to the largest plane and reused. An input the current plane does not have
/// keeps the previous plane's contents: the null pointer in GpuPlaneInput is what says it is absent.
struct GpuPlaneBuffers
{
  DeviceVector<float> cube;
  DeviceVector<float> weights;
  DeviceVector<bool> nonzero_channels;
  DeviceVector<bool> accept_mask;
  DeviceVector<float> selfnoise_offset;
  DeviceVector<float> selfnoise_raw;
  DeviceVector<float> dde_noise;
  DeviceVector<float> mean_image;
  DeviceVector<float> snr_cutoff;
  DeviceVector<float> flux_cutoff;
  DeviceVector<float> baseline_flux;
  DeviceVector<float> beam_kernel; ///< the current band's
};

/// Everything one plane in flight needs on the GPU, reused from plane to plane: the input buffers, the band workspace
/// and outputs, the host staging, and the stream they all run on.
struct GpuSlot
{
  GpuPlaneBuffers inputs;
  GpuBandWorkspace workspace;
  GpuBandResult result;
  GpuBandHostResult host;
  // Declared last: destroyed first, it waits for the work on the buffers above.
  Stream stream;
};
} // namespace breifast::gpu
