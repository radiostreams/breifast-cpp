#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

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

struct GpuBandWorkspaceSizes
{
  std::size_t n_pixels = 0;
  std::size_t stamp_capacity = 0;
  std::size_t stamp_pixels = 0;
  std::size_t beam_stamp_pixels = 0;
  std::size_t histogram_bins = 0;
};

/// The workspace a band needs.
inline GpuBandWorkspaceSizes band_workspace_sizes(const GpuPlaneInput &plane, const GpuBand &band)
{
  const auto stamp_side = static_cast<std::size_t>(2 * (plane.boxsize / 2) + 1);
  const auto beam_side = static_cast<std::size_t>(2 * band.beam_halfsize + 1);
  return GpuBandWorkspaceSizes{
      .n_pixels = plane.n_pixels(),
      .stamp_capacity = std::min(plane.n_pixels(), max_stamp_capacity),
      .stamp_pixels = stamp_side * stamp_side,
      .beam_stamp_pixels = band.beam_kernel != nullptr ? beam_side * beam_side : 0,
      .histogram_bins = static_cast<std::size_t>(2 * plane.max_sigma_bin),
  };
}

/// Device scratch of gpu_median and gpu_median_abs: the 256-bin digit histogram, then the radix-select state (count
/// below the prefix, prefix, coverage) and the largest key below the median.
struct MedianScratch
{
  static constexpr std::size_t histogram_bins = 256;
  DeviceVector<std::uint32_t> words{histogram_bins + 4};
};

/// Device scratch of gpu_label_islands: the per-pixel root ranks, the convergence flag, and the temporary storage of
/// the rank scan (cub::DeviceScan), sized once for n_pixels.
struct LabelScratch
{
  LabelScratch() = default;
  explicit LabelScratch(std::size_t n_pixels); // in gpu_islands.cu, where CUB is available

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
  LabelScratch label_scratch;
  IslandScratch island_scratch;

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
