#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "gpu_kernels.hpp"
#include "gpu_runtime.hpp"

namespace breifast
{
struct PlaneBatchGroup;
struct PlaneConfig;
struct ProcessPlaneResult; // process_plane_result.hpp includes xtensor, which nvcc cannot compile
} // namespace breifast

// The band level of the GPU pipeline, for gpu_plane.cpp and the tests.
namespace breifast::gpu
{
/// The inputs every band of a plane shares, built once per plane: device pointers (null for an absent optional input)
/// and the plane-wide settings.
struct GpuPlaneInput
{
  float *cube = nullptr;             ///< [n_channels, n_y, n_x]; a single-channel band writes its SNR into its channel
  const float *weights = nullptr;    ///< [n_channels]
  const float *mean_image = nullptr; ///< [n_channels, n_y, n_x]; null unless the mean is subtracted
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
  const bool *accept_mask = nullptr;
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
  /// Grows the per-pixel buffers for a plane like `input`; the contents are unspecified after growing.
  void ensure(const GpuPlaneInput &input);

  DeviceVector<float> plane;
  DeviceVector<float> noise_map;
  DeviceVector<float> snr;
  DeviceVector<bool> mask;
  DeviceVector<double> rms_scratch; // gpu_rms_noise_map's scratch
  DeviceVector<float> rms_map;

  DeviceVector<unsigned int> labels;
  LabelScratch label_scratch;
  IslandScratch island_scratch;

  DeviceVector<unsigned int> label_count{1};

  MedianScratch median_scratch;
};

/// Device outputs of gpu_process_band, grown as needed and reused from band to band; the counts say how much of each
/// buffer is valid.
struct GpuBandResult
{
  /// Grows every buffer for `n_islands` islands and `histogram_bins` bins, and sets histogram_bins; the contents are
  /// unspecified after growing.
  void ensure(std::size_t n_islands, std::size_t histogram_bins);

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
/// needs.
void gpu_process_band(const GpuPlaneInput &plane, const GpuBand &band, GpuBandWorkspace &workspace,
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

/// Device copy of one PlaneBatchGroup. An array the group does not have stays empty.
struct GpuGroupInputs
{
  DeviceVector<bool> accept_mask;
  DeviceVector<float> mean_image;
  DeviceVector<float> selfnoise_offset;
  DeviceVector<float> selfnoise_raw;
  DeviceVector<float> dde_noise;
  DeviceVector<float> snr_cutoff;
  DeviceVector<float> flux_cutoff;
  DeviceVector<bool> band_masks;
  std::vector<int> band_accepted;
};

/// What the planes of one call share on the device: every group's arrays and the baseline flux. Kept from call to call
/// and grown as needed.
struct GpuSharedInputs
{
  /// Replaces the contents with `batch_groups` (the mean image only when the config subtracts it) and the config's
  /// baseline flux, and waits for the copies. An input that is absent ends up empty, never left over from before.
  void assign(std::span<const breifast::PlaneBatchGroup> batch_groups, const breifast::PlaneConfig &config);

  std::vector<GpuGroupInputs> groups; // indexed like batch_groups
  DeviceVector<float> baseline_flux;
  DeviceVector<int> band_any;
};

/// Device copies of a plane's own inputs, grown to the largest plane and reused.
struct GpuPlaneBuffers
{
  DeviceVector<float> cube;
  DeviceVector<float> weights;
  DeviceVector<bool> nonzero_channels;
  DeviceVector<float> beam_kernel; ///< the current band's
};

/// Everything one plane in flight needs on the GPU, reused from plane to plane: the input buffers, the band workspace
/// and outputs, the host staging, and the stream they all run on.
struct GpuSlot
{
  Stream stream; // first: destroyed after the buffers
  GpuPlaneBuffers inputs;
  GpuBandWorkspace workspace;
  GpuBandResult result;
  GpuBandHostResult host;
  // Spectral variance moments of the plane, per pixel.
  DeviceVector<float> sum_pow2;
  DeviceVector<float> sum_pow3;
  DeviceVector<float> sum_pow4;

  GpuSlot() = default;
  GpuSlot(const GpuSlot &) = delete;
  GpuSlot(GpuSlot &&) = default;
  GpuSlot &&operator=(const GpuSlot &) = delete;
  GpuSlot &operator=(GpuSlot &&) = default;
};
} // namespace breifast::gpu
