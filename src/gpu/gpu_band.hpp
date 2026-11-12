#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "gpu_kernels.hpp"
#include "gpu_runtime.hpp"

namespace breifast
{
struct ProcessPlaneResult; // process_plane_result.hpp includes xtensor, which nvcc cannot compile
}

// The band level of the GPU pipeline, for gpu_plane.cpp and the tests.
namespace breifast::gpu
{
/// The inputs every band of a plane shares, built once per plane: device pointers (null for an absent optional input)
/// and the plane-wide settings.
struct GpuPlaneInput
{
  const float *cube = nullptr;             ///< [n_channels, n_y, n_x]
  const float *weights = nullptr;          ///< [n_channels]
  const bool *accept_mask = nullptr;       ///< [n_channels, n_y, n_x]
  const float *mean_image = nullptr;       ///< [n_channels, n_y, n_x]; null unless the mean is subtracted
  const float *selfnoise_offset = nullptr; ///< [n_y, n_x]
  const float *selfnoise_raw = nullptr;    ///< [n_y, n_x]
  const float *dde_noise = nullptr;        ///< [n_y, n_x]
  const float *snr_cutoff = nullptr;       ///< [n_channels, n_y, n_x], optional
  const float *flux_cutoff = nullptr;      ///< [n_channels, n_y, n_x], optional
  const float *baseline_flux = nullptr;    ///< [n_y, n_x], optional

  std::size_t n_y = 0;
  std::size_t n_x = 0;
  std::size_t n_channels = 0;

  long boxsize = 0;
  long inner_boxsize = 0;
  long histogram_detail = 0;
  long max_sigma_bin = 0;

  float ddenoise_scaling_factor = 1.F;
  float reject_noise_threshold = 0.F;
  float threshold = 0.F;
  float max_to_std_ratio = 0.F;

  bool use_rms_noise_estimator = false;

  std::size_t n_pixels() const
  {
    return n_y * n_x;
  }
};

/// What changes from one band of the plane to the next.
struct GpuBand
{
  long start = 0;
  long stop = 0;
  float selfnoise_scale_factor = 1.F;
  const float *beam_kernel = nullptr; ///< device, (2 * beam_halfsize + 1)^2; null when the beam is disabled
  long beam_halfsize = 0;
  float beam_kernel_sum = 0.F; ///< summed on the host, as the CPU does
};

/// The per-island outputs of a band, declared once: IslandColumns<DeviceVector> on the device and
/// IslandColumns<std::vector> on the host. tie() lists every column, for the code that treats them all alike.
template <template <class...> class Vector> struct IslandColumns
{
  Vector<float> peak_value;
  Vector<float> peak_local_noise;
  Vector<long> peak_y;
  Vector<long> peak_x;
  Vector<unsigned int> island_size;
  Vector<float> stamp_extreme;
  Vector<float> donut_std;
  Vector<float> donut_lmad;
  Vector<float> beam_chi2;
  Vector<float> flux;
  Vector<float> boost;
  Vector<float> noise_at_peak;
  Vector<float> rms_at_peak;
  Vector<float> selfnoise_at_peak;

  auto tie()
  {
    return std::tie(peak_value, peak_local_noise, peak_y, peak_x, island_size, stamp_extreme, donut_std, donut_lmad,
                    beam_chi2, flux, boost, noise_at_peak, rms_at_peak, selfnoise_at_peak);
  }
  auto tie() const
  {
    return std::tie(peak_value, peak_local_noise, peak_y, peak_x, island_size, stamp_extreme, donut_std, donut_lmad,
                    beam_chi2, flux, boost, noise_at_peak, rms_at_peak, selfnoise_at_peak);
  }
};

/// Device buffers of gpu_process_band, grown to what each band needs and reused from band to band.
struct GpuBandWorkspace
{
  /// Grows every buffer to what `band` of `input` needs; the contents are unspecified after growing.
  void ensure(const GpuPlaneInput &input, const GpuBand &band);

  DeviceVector<float> plane;
  DeviceVector<float> noise_map;
  DeviceVector<float> snr;
  DeviceVector<bool> mask;
  DeviceVector<float> row_sums;
  DeviceVector<float> rms_map;

  DeviceVector<unsigned int> labels;
  LabelScratch label_scratch;
  IslandScratch island_scratch;

  DeviceVector<float> donut_stamps;
  DeviceVector<float> beam_stamps;

  DeviceVector<int> any_flag{1};
  DeviceVector<unsigned int> label_count{1};

  MedianScratch median_scratch;
};

/// Device outputs of gpu_process_band, grown as needed and reused from band to band; the counts say how much of each
/// buffer is valid.
struct GpuBandResult
{
  /// Grows every buffer for `island_capacity` islands and `histogram_bins` bins, and sets histogram_bins; the contents
  /// are unspecified after growing.
  void ensure(std::size_t island_capacity, std::size_t histogram_bins);

  unsigned int label_count = 0;   ///< valid entries of every island column
  std::size_t histogram_bins = 0; ///< valid entries of every histogram

  IslandColumns<DeviceVector> islands;

  DeviceVector<unsigned long long> histogram0;
  DeviceVector<unsigned long long> histogram1;
  DeviceVector<unsigned long long> histogram2;

  DeviceVector<float> min_sigma{1};
  DeviceVector<float> max_sigma{1};
  DeviceVector<float> median_noise{1};
  DeviceVector<float> rms_floor{1};
  DeviceVector<float> global_mad{1};
};

/// Runs one band on `stream` and leaves its outputs in `result`, growing `workspace` and `result` to what the band
/// needs. Returns false when the band's accept mask is empty: the band is skipped, as on the CPU. The caller has
/// already skipped a band with no nonzero channel.
bool gpu_process_band(const GpuPlaneInput &plane, const GpuBand &band, GpuBandWorkspace &workspace,
                      GpuBandResult &result, cudaStream_t stream);

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
void copy_band_result(const GpuBandResult &source, GpuBandHostResult &destination, cudaStream_t stream);

struct GpuBandLabels
{
  std::string stokes_label;
  std::string band_label;
};

/// Adds one processed band to `result` as the CPU process_band does: its histograms replace the previous band's, its
/// median noise is appended, its sigma range merged, and its islands appended as detections. Called in CPU band order.
void append_band_result(const GpuBandHostResult &band, const GpuBandLabels &labels, float max_to_std_ratio,
                        bool use_rms_noise_estimator, breifast::ProcessPlaneResult &result);

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
