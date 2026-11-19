/**
 * @file spectron_plane.hpp
 * @brief C++ port of process_plane() as it stands on breifast's spectron branch (ratt-ru/breifast,
 *        spectron-master-merge): one (interval, Stokes) plane, every band slice, island measurement and the
 *        spectral-variance moments, with the same results as the Python code.
 *
 * The plane loop of upstream breifast (src/plane_lot_dispatch.*) ports the October 2026 master code. Spectron's
 * process_plane has grown a good deal since: usable-band gating, the noise-model and noise-scale terms, Stokes P, the
 * windowed tail calibration, the local flood cap, the edge margin, the rms significance floor, and the PSF-shape and
 * ionospheric-dipole measurements. This file ports that version, statement by statement, including its numerical
 * types: numpy computes a float32 array op in float32 and promotes to float64 only where a float64 operand enters, and
 * the port rounds at the same places, so that thresholds fall the same way.
 *
 * Faults found in the Python code while porting were fixed in both together (2026-11-19: tail calibration counting
 * the acceptance mask, the plane's max sigma overwritten, histograms reset per band slice, NaN medians), so the
 * two paths keep agreeing.
 *
 * The caller (breifast's procres) runs one plane per Python worker thread with the GIL released; nothing here is
 * shared between calls except the max_to_std_ratio cache and the inverse-variance accumulator lock.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace breifast::spectron
{

/// A (band, y, x) array whose planes are C-contiguous; `band_stride` is in elements, 0 for a broadcast band axis.
template <typename T> struct BandArray
{
  const T *data = nullptr;
  long band_stride = 0;
  const T *plane(long band) const
  {
    return data + band * band_stride;
  }
  explicit operator bool() const
  {
    return data != nullptr;
  }
};

/// A float32-or-float64 (band, y, x) array: numpy's dtype decides the arithmetic, so both are kept.
struct BandArrayF
{
  BandArray<float> f32;
  BandArray<double> f64;
  bool is_f64() const
  {
    return static_cast<bool>(f64);
  }
  explicit operator bool() const
  {
    return static_cast<bool>(f32) || static_cast<bool>(f64);
  }
};

/// Per-run settings: the closure variables of process_plane that do not change from plane to plane.
struct Config
{
  long ny = 0, nx = 0, nfreq = 0;
  struct Slice
  {
    long start, stop;
    std::string label;
  };
  std::vector<Slice> band_slices; ///< forward order, as procres builds them

  bool subtract_mean = false;
  enum class NoiseEstimator
  {
    rms,
    mad
  } noise_estimator = NoiseEstimator::rms;
  long boxsize = 0, inner_boxsize = 10;
  double noise_floor_excess = 0.;
  bool selfnoise_bandwidth_scaling = true;
  double ddenoise_scaling_factor = 0.;
  float full_bandwidth = 0.F;
  double reject_noise_threshold = 0.;
  long histogram_detail = 0;
  long max_sigma_bin = 200;
  bool skip_detection = false;
  bool nominal_peaks = false; ///< flood_rate_early_planes and flood_rate_max
  double threshold = 6.;       ///< the nominal detection threshold (tail calibration counts islands above it)

  long flood_cell_max = 0, flood_cell_size = 0, flood_cell_hard = 0;
  long edge_margin = 0;
  double min_sigma_rms = 0.;

  bool beam_size = false;
  bool fixed_beam = false;          ///< beam_kernel is not None
  std::vector<double> fixed_kernel; ///< (2*fixed_halfsize+1)^2
  long fixed_halfsize = 0;
  double fixed_bmaj = 0.; ///< the scalar bmaj of a fixed beam (dipole radius)

  std::vector<double> dip_x, dip_y, dip_flux; ///< empty: no dipole test
  double dipole_radius = 0.;
};

/// Per-plane inputs. Arrays are indexed by the Stokes slot procres uses for this plane (Q's for P).
struct PlaneInputs
{
  // (nfreq, ny, nx) float32 views of the plane's cube slots, modified in place as procres does
  float *cube = nullptr; ///< the slot used for detection (Q's for P)
  long cube_band_stride = 0;
  float *cube_u = nullptr; ///< U for P; when set, the slot above is replaced by sqrt(Q^2 + U^2) first
  long cube_u_band_stride = 0;

  std::vector<uint8_t> nonzero_gate; ///< nfreq
  std::vector<float> weights;        ///< nfreq, float32 as procres has them
  std::vector<float> chanwidth;      ///< nfreq
  BandArray<uint8_t> accept_mask;    ///< (nfreq, ny, nx)
  BandArray<float> mean_image;       ///< (nfreq, ny, nx), when subtract_mean
  BandArrayF snr_cutoff, flux_cutoff;

  // 2D noise terms
  const float *selfnoise_offset_f32 = nullptr;
  const double *selfnoise_offset_f64 = nullptr;
  const float *dde_noise_f32 = nullptr;
  const double *dde_noise_f64 = nullptr;
  const float *nm_noise_f32 = nullptr;
  const double *nm_noise_f64 = nullptr;
  BandArray<double> nm_noise_band; ///< per-band noise model, float64
  const double *noise_scale_sqrt = nullptr;
  const float *selfnoise_f32 = nullptr; ///< raw selfnoise, recorded per detection
  const double *selfnoise_f64 = nullptr;
  const float *baseline_flux = nullptr; ///< float32

  // per-plane beam (float32, one per band), when beam_size and no fixed beam
  std::vector<float> bmaj, bmin, bpa;

  std::vector<double> slice_threshold; ///< detection threshold in force per band slice (forward order)

  // accumulator for the per-band inverse thermal variance (pb_ivar_sum[stokes]), shared between planes
  double *pb_ivar = nullptr;
  std::mutex *pb_ivar_lock = nullptr;

  /// procres' spectral-variance accumulators, folded here (what process_plane_result does with variance_stats) so
  /// that procres' main thread does not have to: per-Stokes slices except the maxima, which take every slot
  struct Accumulators
  {
    double *spvar_sum = nullptr, *spskew_sum = nullptr, *spkurt_sum = nullptr; ///< (ny, nx)
    double *spinvvar_sum = nullptr, *spmodel_sum = nullptr;                    ///< (ny, nx)
    int64_t *spvar_counts = nullptr;                                           ///< (ny, nx)
    double *spvar_max = nullptr, *spkurt_max = nullptr;                        ///< (nslots, ny, nx)
    long nslots = 0;
    double *pb_var_sum = nullptr;     ///< (nfreq, ny, nx)
    int64_t *pb_var_counts = nullptr; ///< (nfreq)
    std::mutex *lock = nullptr;
  };
  std::optional<Accumulators> accumulate;
};

/// What process_plane returns, minus what the caller already has.
struct PlaneResult
{
  bool returned_early = false; ///< no usable band: process_plane returned before the band loop
  std::vector<uint8_t> usable;
  long live_without_finite = 0;

  bool any_minsig = false;
  double minsig = 0., maxsig = 0.;
  std::optional<std::vector<long>> hist[3];
  std::vector<double> mednoises;
  double last_mednoise = 0.; ///< the `mednoise` the detection table divides by (the last band's)

  // detections
  std::vector<long> x, y, size, flooded;
  std::vector<std::string> band;
  std::vector<float> flux, sigma_gmad, selfnoise, sigma_rms, sigma_lmad;
  std::vector<double> sigma, sigma_ext, enf, beamchi2, psf_iso, psf_iso_x, psf_corr, dip_src, dip_mirror, dip_sep,
      localnoise, boost;

  // nominal-threshold islands, by band label: (peak |sigma|, y, x) per island, flattened, the position being the
  // first pixel in raster order at the peak (the tail calibration counts distinct positions)
  std::vector<std::pair<std::string, std::vector<double>>> nominal_peaks;

  // counters the caller adds to its own
  long flood_dropped_islands = 0, flood_dropped_cells = 0, edge_dropped = 0, rms_dropped = 0, flood_tagged = 0;

  // spectral variance (nfreq > 1)
  bool has_variance = false;
  std::vector<float> spvar, spskew, spkurt; ///< (ny, nx)
  long counts = 0;
  std::vector<float> pb_var;       ///< (nfreq, ny, nx)
  std::vector<double> inv_var;     ///< (ny, nx), empty when no single-band slice contributed
  std::vector<double> model_excess;
};

/**
 * @brief Process one plane exactly as spectron's process_plane does, from the point where the cube slice is loaded
 *        and the plane known to be gated in (the caller does the nonzero gate and the loading).
 */
void process_plane(const Config &config, PlaneInputs &in, PlaneResult &out);

/// breifast.max_to_std_ratio(n), cached.
double max_to_std_ratio(long n);

} // namespace breifast::spectron
