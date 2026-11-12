#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "device_vector.hpp"
#include "gpu_kernels.hpp"

namespace breifast::gpu
{
inline constexpr std::size_t max_stamp_capacity = 4096;

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

/// Device scratch of gpu_median and gpu_median_abs: the 256-bin digit histogram, then the radix-select state (count
/// below the prefix, prefix, coverage) and the largest key below the median.
struct MedianScratch
{
  static constexpr std::size_t histogram_bins = 256;
  DeviceVector<std::uint32_t> words{histogram_bins + 4};
};

/// Device scratch of gpu_label_islands: the per-pixel root ranks, the convergence flag, and the temporary storage of
/// the rank scan (cub::DeviceScan).
struct LabelScratch
{
  /// Grows every array for a plane of `n_pixels`; the contents are unspecified after growing.
  void ensure(std::size_t n_pixels); // in gpu_islands.cu, where CUB is available

  DeviceVector<unsigned int> root_ranks;
  DeviceVector<unsigned int> changed{1};
  DeviceVector<std::byte> scan_storage;
};

/// Device scratch of gpu_label_stats and gpu_island_boost: the per-island reduction keys and partial results.
struct IslandScratch
{
  /// Grows every array to hold `capacity` islands; the contents are unspecified after growing.
  void ensure(std::size_t capacity)
  {
    value_keys.ensure(capacity);
    local_noise_max.ensure(capacity);
    sizes.ensure(capacity);
    boost_keys.ensure(capacity + 1); // indexed by label, 1..capacity
  }

  DeviceVector<unsigned long long> value_keys;
  DeviceVector<float> local_noise_max;
  DeviceVector<unsigned int> sizes;
  DeviceVector<unsigned long long> boost_keys;
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
} // namespace breifast::gpu
