#pragma once

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_reduce.h"
#include <chrono>
#include <iostream>
#define XTENSOR_USE_TBB
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/reducers/xreducer.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>

/**
 * @brief Builds a 2D summed-area table (integral image) of `img`, for O(1) box-sum queries via box_sum().
 *
 * `integral(i, j) = sum of img(0..i-1, 0..j-1)`. The extra leading zero row/column lets box_sum() compute any
 * rectangle's sum by inclusion-exclusion (`integral(r1,c1) - integral(r0,c1) - integral(r1,c0) + integral(r0,c0)`)
 * without needing to special-case rectangles touching row/column 0.
 *
 * `Accum` (defaulted to `double`) is the accumulation type, independent of `img`'s own (always float32) storage.
 * A box sum here can cover tens to hundreds of thousands of pixels (boxsize up to a few hundred), and summing
 * that many float32 values in `Accum = float` loses enough precision to measurably shift the noise map --
 * confirmed by comparing against the pure-Python reference, where staying in float32 throughout pushed ~26% of a
 * real image's SNR values outside the regression tolerance. `Accum = float` is kept available (rather than
 * hardcoding double) specifically so make_noise_map()'s two instantiations can be A/B compared for
 * performance/precision -- see bench.cpp.
 *
 * @tparam Accum Accumulation type (`double` by default).
 * @param img (H, W) input image.
 * @return (H+1, W+1) summed-area table, in `Accum` precision.
 */
template <typename Accum = double> inline xt::xarray<Accum> compute_integral_image(const xt::xarray<float> &img)
{
  const size_t h = img.shape(0), w = img.shape(1);
  xt::xarray<Accum> integral = xt::zeros<Accum>({h + 1, w + 1});
  for (size_t i = 0; i < h; ++i)
  {
    Accum row_sum = Accum{0};
    for (size_t j = 0; j < w; ++j)
    {
      row_sum += static_cast<Accum>(img(i, j));
      integral(i + 1, j + 1) = integral(i, j + 1) + row_sum;
    }
  }
  return integral;
}

/**
 * @brief Sum of the original image's values over the rectangle [r0,r1) x [c0,c1), clamped to the image bounds.
 *
 * Inclusion-exclusion on the summed-area table from compute_integral_image(). Clamping lets callers pass windows
 * that extend past the image edge (e.g. a sliding box near the border) without bounds-checking themselves -- the
 * out-of-bounds portion simply contributes nothing, i.e. zero-padding.
 *
 * @tparam Accum Accumulation type; must match the `integral` table's own element type.
 * @param integral (h+1, w+1) summed-area table from compute_integral_image<Accum>().
 * @param r0,r1 Row range [r0, r1), before clamping to [0, h).
 * @param c0,c1 Column range [c0, c1), before clamping to [0, w).
 * @param h,w Original image dimensions.
 * @return Sum over the clamped rectangle; 0 if it's empty after clamping.
 */
template <typename Accum = double>
inline Accum box_sum(const xt::xarray<Accum> &integral, long r0, long r1, long c0, long c1, long h, long w)
{
  r0 = std::clamp(r0, 0L, h);
  r1 = std::clamp(r1, 0L, h);
  c0 = std::clamp(c0, 0L, w);
  c1 = std::clamp(c1, 0L, w);
  if (r0 >= r1 || c0 >= c1)
  {
    return Accum{0};
  }
  return integral(r1, c1) - integral(r0, c1) - integral(r1, c0) + integral(r0, c0);
}

/**
 * @brief Sliding-window local RMS noise map for `plane_img`, over a `boxsize` x `boxsize` box.
 *
 * Matches `scipy.ndimage.uniform_filter(mode='constant')` on `plane_img^2`, corrected for partial boxes at the
 * edges (normalized by the true in-bounds pixel count, not the full box area). Noise is floored at the image's
 * median to avoid spuriously low values. Port of breifast.py's make_noise_map(..., method='std').
 *
 * Box sums are accumulated in `Accum` precision (see compute_integral_image()); only the final per-pixel result
 * is cast back to float32. Production code should use the `Accum = double` default -- see that default's own
 * docstring for why. `Accum = float` exists for A/B performance/precision comparison only.
 *
 * @tparam Accum Accumulation type (`double` by default).
 * @param plane_img (H, W) image to estimate noise for.
 * @param boxsize Side length of the sliding box.
 * @return {(H, W) noise map, its median}.
 */
template <typename Accum = double>
inline std::pair<xt::xarray<float>, float> make_noise_map(const xt::xarray<float> &plane_img, long boxsize)
{
  const long h = static_cast<long>(plane_img.shape(0));
  const long w = static_cast<long>(plane_img.shape(1));
  const long lo = boxsize / 2;
  const Accum box_area = static_cast<Accum>(boxsize) * static_cast<Accum>(boxsize);

  const xt::xarray<float> img_sq = plane_img * plane_img;
  const auto integral_sq = compute_integral_image<Accum>(img_sq);

  xt::xarray<float> noise = xt::zeros<float>({static_cast<size_t>(h), static_cast<size_t>(w)});

  for (long i = 0; i < h; ++i)
  {
    const long r0 = i - lo, r1 = r0 + boxsize;
    for (long j = 0; j < w; ++j)
    {
      const long c0 = j - lo, c1 = c0 + boxsize;
      const Accum sum2 = box_sum<Accum>(integral_sq, r0, r1, c0, c1, h, w);
      const Accum valid_count = static_cast<Accum>(std::max(0L, std::min(r1, h) - std::max(r0, 0L))) *
                                static_cast<Accum>(std::max(0L, std::min(c1, w) - std::max(c0, 0L)));
      const Accum scale = std::sqrt(valid_count / box_area);
      const Accum raw_noise = std::sqrt(sum2 / box_area);
      noise(i, j) = scale != Accum{0} ? static_cast<float>(raw_noise / scale) : 0.f;
    }
  }

  const float median_noise = xt::median(noise);
  xt::filter(noise, noise < median_noise) = median_noise;

  return {noise, median_noise};
}

/**
 * @brief Fixed-width histogram of `values` at masked pixels, matching `np.histogram(values[mask],
 * np.arange(-max_sigma_bin, max_sigma_bin + 1))`.
 *
 * Unit-width integer bins over [-max_sigma_bin, max_sigma_bin]; values outside that range are dropped, not
 * clamped into the edge bins.
 *
 * @param values Values to bin (e.g. SNR).
 * @param mask Which pixels of `values` to include.
 * @param max_sigma_bin Histogram half-range; produces `2 * max_sigma_bin` bins.
 * @return Per-bin counts, length `2 * max_sigma_bin`.
 */
inline std::vector<long> histogram_counts(const xt::xarray<float> &values, const xt::xarray<bool> &mask,
                                          int max_sigma_bin)
{
  const int nbins = 2 * max_sigma_bin;
  std::vector<long> counts(nbins, 0);
  auto it_v = values.begin();
  auto it_m = mask.begin();
  for (; it_v != values.end(); ++it_v, ++it_m)
  {
    if (!*it_m)
    {
      continue;
    }
    const float v = *it_v;
    if (v < -max_sigma_bin || v > max_sigma_bin)
    {
      continue;
    }
    int idx = static_cast<int>(std::floor(v + max_sigma_bin));
    if (idx >= nbins)
    {
      idx = nbins - 1;
    }
    if (idx < 0)
    {
      idx = 0;
    }
    counts[idx]++;
  }
  return counts;
}

// Result of estimate_noise_and_snr(): the noise map, its median and excess factor, plus the resulting SNR image
// and validity mask.
struct NoiseMapResult
{
  xt::xarray<float> noisemap;
  xt::xarray<float> excess_noise_factor;
  float mednoise;
  xt::xarray<float> snr;
  xt::xarray<float> abs_snr;
  xt::xarray<bool> valid_noisemap;
};

/**
 * @brief Builds the per-pixel noise map for a band and converts `plane_img` to SNR.
 *
 * `noisemap0` is either the sliding RMS map (make_noise_map()) or a flat map of `noise_globmad`. The final
 * noise map combines `noisemap0` in quadrature with two extra variance terms: `sqrt(noisemap0^2 + snsf *
 * selfnoise_offset + ddenoise_scaling_factor * dde_noise)`. `snr = plane_img / noisemap`, zeroed (and excluded via
 * `valid_noisemap`) wherever `noisemap` is zero or non-finite.
 *
 * @param plane_img (H, W) flux image for this band.
 * @param use_rms_noise_estimator `true` for the sliding RMS map, `false` for a flat map of `noise_globmad`.
 * @param boxsize Box side length for the RMS estimator.
 * @param noise_globmad Global MAD noise estimate, used directly when `use_rms_noise_estimator` is false.
 * @param snsf Scale factor applied to `selfnoise_offset`.
 * @param selfnoise_offset (H, W) selfnoise variance term.
 * @param dde_noise (H, W) DDE-calibration noise variance term.
 * @param ddenoise_scaling_factor Scale factor applied to `dde_noise`.
 * @return See NoiseMapResult.
 */
inline NoiseMapResult estimate_noise_and_snr(const xt::xarray<float> &plane_img, bool use_rms_noise_estimator,
                                             int boxsize, float noise_globmad, float snsf,
                                             const xt::xarray<float> &selfnoise_offset,
                                             const xt::xarray<float> &dde_noise, float ddenoise_scaling_factor)
{
  xt::xarray<float> noisemap0;
  if (use_rms_noise_estimator)
  {
    auto result = make_noise_map(plane_img, boxsize);
    noisemap0 = std::move(result.first);
  }
  else
  {
    noisemap0 = xt::full_like(plane_img, noise_globmad);
  }

  xt::xarray<float> noisemap =
      xt::sqrt(noisemap0 * noisemap0 + snsf * selfnoise_offset + ddenoise_scaling_factor * dde_noise);
  xt::xarray<float> excess_noise_factor = xt::where(xt::not_equal(noisemap0, 0.f), noisemap / noisemap0, 0.f);
  const float mednoise = static_cast<float>(xt::median(noisemap));

  xt::xarray<float> snr = xt::where(xt::not_equal(noisemap, 0.f), plane_img / noisemap, 0.f);
  xt::xarray<bool> valid_noisemap = xt::not_equal(noisemap, 0.f) & xt::isfinite(noisemap);
  snr = xt::where(valid_noisemap, snr, 0.f);
  xt::xarray<float> abs_snr = xt::abs(snr);

  return {std::move(noisemap), std::move(excess_noise_factor), mednoise, std::move(snr),
          std::move(abs_snr),  std::move(valid_noisemap)};
}

// Result of apply_rejection_cutoffs(): the narrowed image_mask, plus hist1 (after the SNR cutoff) and hist2 (after
// the flux cutoff). hist0, taken before any of these run, stays in process_band_slice_core() since it needs
// nothing this function computes.
struct CutoffResult
{
  xt::xarray<bool> image_mask;
  std::vector<long> hist1;
  std::vector<long> hist2;
};

/**
 * @brief Narrows `image_mask` through the SNR, flux, and local-noise rejection cutoffs, capturing hist1/hist2
 * along the way.
 *
 * Each cutoff is skippable: SNR/flux cutoffs are skipped if their cutoff image is absent, the local-noise cutoff
 * is skipped if `reject_noise_threshold` is 0. `snr_cutoff`/`flux_cutoff` are averaged over the band before
 * comparison; the local-noise cutoff rejects pixels where `noisemap > mednoise * reject_noise_threshold`.
 *
 * @param image_mask Mask to narrow (taken by value, returned via CutoffResult).
 * @param snr Per-pixel SNR image, for the histograms.
 * @param abs_snr |SNR|, compared against the SNR cutoff.
 * @param plane_img Flux image, compared against the flux cutoff.
 * @param noisemap Per-pixel noise map, compared against the local-noise cutoff.
 * @param mednoise Median of `noisemap`, scaled by `reject_noise_threshold`.
 * @param snr_cutoff (FREQ, H, W) per-pixel SNR cutoff image, or absent to skip.
 * @param flux_cutoff (FREQ, H, W) per-pixel flux cutoff image, or absent to skip.
 * @param reject_noise_threshold Local-noise cutoff threshold as a multiple of `mednoise`; 0 disables it.
 * @param band_start,band_stop This band's channel range, for averaging the cutoff images over the band.
 * @param band_length Number of channels in the band (`band_stop - band_start`).
 * @param histogram_detail Controls which of hist1/hist2 get computed (see CutoffResult).
 * @param max_sigma_bin Histogram half-range in sigma.
 * @return See CutoffResult.
 */
inline CutoffResult apply_rejection_cutoffs(xt::xarray<bool> image_mask, const xt::xarray<float> &snr,
                                            const xt::xarray<float> &abs_snr, const xt::xarray<float> &plane_img,
                                            const xt::xarray<float> &noisemap, float mednoise,
                                            const std::optional<xt::xarray<float>> &snr_cutoff,
                                            const std::optional<xt::xarray<float>> &flux_cutoff,
                                            float reject_noise_threshold, long band_start, long band_stop,
                                            long band_length, int histogram_detail, int max_sigma_bin)
{
  std::vector<long> hist1, hist2;

  if (snr_cutoff.has_value())
  {
    const xt::xarray<float> band_snr_cutoff_sum =
        xt::sum(xt::xarray<float>(xt::view(*snr_cutoff, xt::range(band_start, band_stop), xt::all(), xt::all())), {0});
    const xt::xarray<float> band_snr_cutoff = band_snr_cutoff_sum / static_cast<float>(band_length);
    image_mask = image_mask & (abs_snr >= band_snr_cutoff);
  }

  if (histogram_detail > 1)
    hist1 = histogram_counts(snr, image_mask, max_sigma_bin);

  if (flux_cutoff.has_value())
  {
    const xt::xarray<float> band_flux_cutoff_sum =
        xt::sum(xt::xarray<float>(xt::view(*flux_cutoff, xt::range(band_start, band_stop), xt::all(), xt::all())), {0});
    const xt::xarray<float> band_flux_cutoff = band_flux_cutoff_sum / static_cast<float>(band_length);
    image_mask = image_mask & (xt::abs(plane_img) >= band_flux_cutoff);
  }

  if (histogram_detail > 0)
    hist2 = histogram_counts(snr, image_mask, max_sigma_bin);

  if (reject_noise_threshold != 0.f)
  {
    image_mask = image_mask & (noisemap <= mednoise * reject_noise_threshold);
  }

  return {std::move(image_mask), std::move(hist1), std::move(hist2)};
}

// Result of process_band_slice_core(). Mirrors the 13-element tuple the pybind11-facing
// process_band_slice() returns, minus the Python marshaling: `processed == false` means the same
// "nothing to report" case that returns all-None fields there (mednoise/noise_globmad are still
// set if the plane got past the first early-exit but not the second -- see process_band_slice_core()).
struct ProcessBandSliceResult
{
  bool processed = false;
  bool reached_cutoffs = false;
  xt::xarray<float> snr;
  xt::xarray<bool> mask;
  xt::xarray<float> plane_img;
  xt::xarray<float> noisemap;
  xt::xarray<float> excess_noise_factor;
  float mednoise = 0.f;
  float noise_globmad = 0.f;
  float minsig = 0.f;
  float maxsig = 0.f;
  std::vector<long> hist0, hist1, hist2;
};

/**
 * @brief Computes the per-pixel SNR image and detection mask for one band slice of one Stokes plane.
 *
 * Port of the per-band-slice body of process_plane(). Python/pybind11-independent core shared by the
 * process_band_slice() pybind11 wrapper (breifast_helpers.cpp) and the standalone profiling driver
 * (bench.cpp) -- see that file's docstring for the algorithm description.
 *
 * @param stokes_cube (FREQ, H, W) flux cube for this Stokes/interval. Mutated in place for single-channel bands
 *        (overwritten with `snr`, so aggregated bands built from this channel later see SNR, not flux).
 * @param weights (FREQ,) per-channel weights for this Stokes/interval.
 * @param accept_mask (FREQ, H, W) per-pixel acceptance mask for this Stokes.
 * @param channel_width (FREQ,) full per-channel bandwidths.
 * @param band_start,band_stop Channel range [band_start, band_stop) making up this band.
 * @param is_single_channel Whether this band covers exactly one channel.
 * @param subtract_mean Whether to subtract the band-averaged mean image (long-term static-source model) first, to
 *        isolate the transient/variable signal.
 * @param mean_image (FREQ, H, W) mean image for this Stokes, or absent if `subtract_mean` is false.
 * @param use_rms_noise_estimator `true` for the sliding-window RMS noise estimator, `false` for the global MAD
 *        estimate.
 * @param boxsize Side length of the sliding box used by the RMS noise estimator.
 * @param selfnoise_offset (H, W) per-pixel selfnoise variance term.
 * @param dde_noise (H, W) per-pixel DDE-calibration noise variance term.
 * @param ddenoise_scaling_factor Scale factor applied to `dde_noise`.
 * @param full_bandwidth Total observing bandwidth across all channels.
 * @param flux_cutoff (FREQ, H, W) per-pixel flux cutoff image, or absent to skip.
 * @param snr_cutoff (FREQ, H, W) per-pixel SNR cutoff image, or absent to skip.
 * @param reject_noise_threshold Local-noise rejection threshold as a multiple of `mednoise`; 0 disables it.
 * @param histogram_detail Controls which of hist0/hist1/hist2 get computed (see ProcessBandSliceResult).
 * @param max_sigma_bin Histogram half-range in sigma; unit-width bins over [-max_sigma_bin, max_sigma_bin].
 * @param threshold Final detection threshold on |SNR|.
 * @return See ProcessBandSliceResult.
 */
inline ProcessBandSliceResult process_band_slice_core(
    xt::xarray<float> &stokes_cube, const xt::xarray<float> &weights, const xt::xarray<bool> &accept_mask,
    const xt::xarray<float> &channel_width, long band_start, long band_stop, bool is_single_channel, bool subtract_mean,
    const std::optional<xt::xarray<float>> &mean_image, bool use_rms_noise_estimator, int boxsize,
    const xt::xarray<float> &selfnoise_offset, const xt::xarray<float> &dde_noise, float ddenoise_scaling_factor,
    float full_bandwidth, const std::optional<xt::xarray<float>> &flux_cutoff,
    const std::optional<xt::xarray<float>> &snr_cutoff, float reject_noise_threshold, int histogram_detail,
    int max_sigma_bin, float threshold)
{
  const long band_length = band_stop - band_start;

  const xt::xarray<float> ww = xt::view(weights, xt::range(band_start, band_stop));
  const float sumww = xt::sum(ww)();

  xt::xarray<float> plane_img;
  if (is_single_channel)
  {
    plane_img = xt::view(stokes_cube, band_start, xt::all(), xt::all());
  }
  else
  {
    const xt::xarray<float> band = xt::view(stokes_cube, xt::range(band_start, band_stop), xt::all(), xt::all());
    const auto weighted = band * xt::view(ww, xt::all(), xt::newaxis(), xt::newaxis());
    plane_img = xt::sum(weighted, {0}) / sumww;
  }

  const xt::xarray<bool> band_accept_mask =
      xt::view(accept_mask, xt::range(band_start, band_stop), xt::all(), xt::all());
  xt::xarray<bool> image_mask = xt::reduce(std::logical_or<bool>(), band_accept_mask, {0});

  // Early exit for empty images.
  const bool empty_plane = xt::all(xt::equal(plane_img, 0.f));
  if (empty_plane || !xt::any(image_mask))
  {
    return {};
  }

  // Optional mean substraction to only keep dynamics.
  if (subtract_mean && mean_image.has_value())
  {
    const xt::xarray<float> band_mean = xt::view(*mean_image, xt::range(band_start, band_stop), xt::all(), xt::all());
    plane_img = plane_img - xt::sum(band_mean, {0}) / static_cast<float>(band_mean.shape(0));
  }

  // SNSF: narrower bands get less spectral averaging, so more noise -- inflates selfnoise below.
  const float bandwidth = xt::sum(xt::xarray<float>(xt::view(channel_width, xt::range(band_start, band_stop))))();
  const float snsf = full_bandwidth / bandwidth;
  const xt::xarray<float> abs_plane_img = xt::abs(plane_img);
  const float noise_globmad = xt::median(abs_plane_img) * 1.4826f;

  // Divide plane_img (flux) by this band's estimated noise to get SNR -- flux alone isn't comparable across
  // pixels/bands, SNR is
  NoiseMapResult noise_result = estimate_noise_and_snr(plane_img, use_rms_noise_estimator, boxsize, noise_globmad, snsf,
                                                       selfnoise_offset, dde_noise, ddenoise_scaling_factor);
  xt::xarray<float> noisemap = std::move(noise_result.noisemap);
  xt::xarray<float> excess_noise_factor = std::move(noise_result.excess_noise_factor);
  const float mednoise = noise_result.mednoise;
  xt::xarray<float> snr = std::move(noise_result.snr);
  xt::xarray<float> abs_snr = std::move(noise_result.abs_snr);

  image_mask = image_mask & noise_result.valid_noisemap;

  std::vector<long> hist0;
  if (histogram_detail > 1)
  {
    hist0 = histogram_counts(snr, image_mask, max_sigma_bin);
  }

  // Reject pixels that are too weak in SNR, too weak in flux, or in too noisy a region -- each cutoff narrows
  // image_mask
  CutoffResult cutoffs = apply_rejection_cutoffs(std::move(image_mask), snr, abs_snr, plane_img, noisemap, mednoise,
                                                 snr_cutoff, flux_cutoff, reject_noise_threshold, band_start, band_stop,
                                                 band_length, histogram_detail, max_sigma_bin);
  image_mask = std::move(cutoffs.image_mask);
  std::vector<long> hist1 = std::move(cutoffs.hist1);
  std::vector<long> hist2 = std::move(cutoffs.hist2);

  // Early exit if all pixels are masked
  if (!xt::any(image_mask))
  {
    ProcessBandSliceResult result;
    result.processed = false;
    result.reached_cutoffs = true;
    result.mednoise = mednoise;
    result.noise_globmad = noise_globmad;
    result.hist0 = std::move(hist0);
    result.hist1 = std::move(hist1);
    result.hist2 = std::move(hist2);
    return result;
  }

  float minsig = std::numeric_limits<float>::infinity();
  float maxsig = -std::numeric_limits<float>::infinity();
  {
    auto it_v = snr.begin();
    auto it_m = image_mask.begin();
    for (; it_v != snr.end(); ++it_v, ++it_m)
    {
      if (!*it_m)
      {
        continue;
      }
      minsig = std::min(minsig, *it_v);
      maxsig = std::max(maxsig, *it_v);
    }
  }

  abs_snr = xt::where(image_mask, abs_snr, 0.f);
  xt::xarray<bool> final_mask = abs_snr >= threshold;

  if (is_single_channel)
  {
    xt::view(stokes_cube, band_start, xt::all(), xt::all()) = snr;
  }

  ProcessBandSliceResult result;
  result.processed = true;
  result.reached_cutoffs = true;
  result.snr = std::move(snr);
  result.mask = std::move(final_mask);
  result.plane_img = std::move(plane_img);
  result.noisemap = std::move(noisemap);
  result.excess_noise_factor = std::move(excess_noise_factor);
  result.mednoise = mednoise;
  result.noise_globmad = noise_globmad;
  result.minsig = minsig;
  result.maxsig = maxsig;
  result.hist0 = std::move(hist0);
  result.hist1 = std::move(hist1);
  result.hist2 = std::move(hist2);
  return result;
}

/**
 * @brief RAII timer that accumulates elapsed time per name across every instance sharing that name.
 */
class ScopedTimer
{
public:
  struct Stats
  {
    long count = 0;
    double total_seconds = 0.0;
    double min_seconds = std::numeric_limits<double>::infinity();
    double max_seconds = 0.0;
  };

  explicit ScopedTimer(std::string name) : name_(std::move(name)), begin_(std::chrono::high_resolution_clock::now())
  {
  }

  ~ScopedTimer()
  {
    const auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin_).count();
    std::lock_guard<std::mutex> lock(mutex());
    Stats &stats = registry()[name_];
    ++stats.count;
    stats.total_seconds += seconds;
    stats.min_seconds = std::min(stats.min_seconds, seconds);
    stats.max_seconds = std::max(stats.max_seconds, seconds);
  }

  static void report(std::ostream &os = std::cout)
  {
    std::lock_guard<std::mutex> lock(mutex());
    for (const auto &[name, stats] : registry())
    {
      os << name << ": count=" << stats.count << " total=" << stats.total_seconds << "s"
         << " mean=" << (stats.total_seconds / static_cast<double>(stats.count)) << "s"
         << " min=" << stats.min_seconds << "s"
         << " max=" << stats.max_seconds << "s" << std::endl;
    }
  }

  static void reset()
  {
    std::lock_guard<std::mutex> lock(mutex());
    registry().clear();
  }

private:
  static std::mutex &mutex()
  {
    static std::mutex m;
    return m;
  }

  static std::unordered_map<std::string, Stats> &registry()
  {
    static std::unordered_map<std::string, Stats> r;
    return r;
  }

  std::string name_;
  std::chrono::high_resolution_clock::time_point begin_;
};

/**
 * @brief Median of `values` (numpy.median convention: average of two middle values when even).
 *
 * nth_element on a copy, not a full sort -- avoids the median-of-a-lazy-expression hotspot flagged in
 * process_band_slice_core()'s docstring.
 */
inline float median(std::vector<float> values)
{
  const size_t n = values.size();
  const size_t mid = n / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
  const float upper = values[mid];
  if (n % 2 == 1)
  {
    return upper;
  }
  const float lower = *std::max_element(values.begin(), values.begin() + static_cast<long>(mid));
  return 0.5f * (upper + lower);
}

/**
 * @brief Raw-pointer overload of make_noise_map(): same algorithm over flat row-major std::vector
 * storage instead of xt::xarray, to keep process_band_slice_core_serial() xtensor-free.
 *
 * @param plane_img (H*W) row-major flattened image.
 */
inline std::vector<float> make_noise_map(const std::vector<float> &plane_img, size_t height, size_t width, long boxsize)
{
  const long h = static_cast<long>(height);
  const long w = static_cast<long>(width);
  const long lo = boxsize / 2;
  const double box_area = static_cast<double>(boxsize) * static_cast<double>(boxsize);
  const size_t istride = static_cast<size_t>(w + 1);

  std::vector<double> integral(static_cast<size_t>(h + 1) * istride, 0.0);
  for (size_t i = 0; i < height; ++i)
  {
    double row_sum = 0.0;
    const size_t src_row = i * width;
    const size_t dst_row = (i + 1) * istride;
    const size_t prev_row = i * istride;
    for (size_t j = 0; j < width; ++j)
    {
      const float v = plane_img[src_row + j];
      row_sum += static_cast<double>(v) * v;
      integral[dst_row + j + 1] = integral[prev_row + j + 1] + row_sum;
    }
  }

  const auto box_sum2 = [&](long r0, long r1, long c0, long c1) -> double
  {
    r0 = std::clamp(r0, 0L, h);
    r1 = std::clamp(r1, 0L, h);
    c0 = std::clamp(c0, 0L, w);
    c1 = std::clamp(c1, 0L, w);
    if (r0 >= r1 || c0 >= c1)
    {
      return 0.0;
    }
    return integral[static_cast<size_t>(r1) * istride + static_cast<size_t>(c1)] -
           integral[static_cast<size_t>(r0) * istride + static_cast<size_t>(c1)] -
           integral[static_cast<size_t>(r1) * istride + static_cast<size_t>(c0)] +
           integral[static_cast<size_t>(r0) * istride + static_cast<size_t>(c0)];
  };

  std::vector<float> noise(height * width);
  for (size_t i = 0; i < height; ++i)
  {
    const long r0 = static_cast<long>(i) - lo, r1 = r0 + boxsize;
    const size_t row = i * width;
    for (size_t j = 0; j < width; ++j)
    {
      const long c0 = static_cast<long>(j) - lo, c1 = c0 + boxsize;
      const double sum2 = box_sum2(r0, r1, c0, c1);
      const double valid_count = static_cast<double>(std::max(0L, std::min(r1, h) - std::max(r0, 0L))) *
                                 static_cast<double>(std::max(0L, std::min(c1, w) - std::max(c0, 0L)));
      const double scale = std::sqrt(valid_count / box_area);
      const double raw_noise = std::sqrt(sum2 / box_area);
      noise[row + j] = scale != 0.0 ? static_cast<float>(raw_noise / scale) : 0.f;
    }
  }

  const float median_noise = median(noise);
  for (float &v : noise)
  {
    if (v < median_noise)
    {
      v = median_noise;
    }
  }
  return noise;
}

/**
 * @brief Hand-rolled, xtensor-free reimplementation of process_band_slice_core() -- see TODO.md item 4.
 *
 * Four full-array passes (see the Pass N comments below), split only where a median or a fully-built
 * array is needed before the next step. Checked against process_band_slice_core() in
 * tests/cpp/test_breifast_core.cpp; xt::xarray/std::optional parameters stay as-is for interop with
 * process_plane_core() and the pybind11 wrapper.
 */
inline ProcessBandSliceResult process_band_slice_core_serial(
    xt::xarray<float> &stokes_cube_arr, const xt::xarray<float> &weights, const xt::xarray<bool> &accept_mask,
    const xt::xarray<float> &channel_width, long band_start, long band_stop, bool is_single_channel, bool subtract_mean,
    const std::optional<xt::xarray<float>> &mean_image, bool use_rms_noise_estimator, int boxsize,
    const xt::xarray<float> &selfnoise_offset, const xt::xarray<float> &dde_noise, float ddenoise_scaling_factor,
    float full_bandwidth, const std::optional<xt::xarray<float>> &flux_cutoff,
    const std::optional<xt::xarray<float>> &snr_cutoff, float reject_noise_threshold, int histogram_detail,
    int max_sigma_bin, float threshold)
{
  assert(band_stop > band_start);
  const size_t band_length = band_stop - band_start;
  const size_t height = stokes_cube_arr.shape(1);
  const size_t width = stokes_cube_arr.shape(2);
  const size_t npix = height * width;

  float *stokes_cube = stokes_cube_arr.data();
  const bool *accept_mask_data = accept_mask.data();

  std::vector<float> ww(band_length);
  float sumww = 0.f;
  for (size_t f = 0; f < band_length; ++f)
  {
    ww[f] = weights(band_start + static_cast<long>(f));
    sumww += ww[f];
  }

  const int nbins = 2 * max_sigma_bin;
  auto bin_into = [max_sigma_bin, nbins](std::vector<long> &hist, float v)
  {
    if (v < -max_sigma_bin || v > max_sigma_bin)
    {
      return;
    }
    const int idx = std::clamp(static_cast<int>(std::floor(v + max_sigma_bin)), 0, nbins - 1);
    ++hist[static_cast<size_t>(idx)];
  };

  // Pass 1: plane_img + image_mask + early-exit flags.
  std::vector<float> plane_img(npix);
  std::vector<uint8_t> image_mask(npix);
  bool any_nonzero = false;
  bool any_masked = false;
  if (is_single_channel)
  {
    const size_t base = static_cast<size_t>(band_start) * npix;
    for (size_t iy = 0; iy < height; ++iy)
    {
      for (size_t ix = 0; ix < width; ++ix)
      {
        const size_t p = iy * width + ix;
        const float v = stokes_cube[base + p];
        const bool m = accept_mask_data[base + p];
        plane_img[p] = v;
        image_mask[p] = m;
        any_nonzero = any_nonzero || (v != 0.f);
        any_masked = any_masked || m;
      }
    }
  }
  else
  {
    // Channel outermost: sequential reads instead of an npix stride per channel.
    std::fill(plane_img.begin(), plane_img.end(), 0.f);
    std::fill(image_mask.begin(), image_mask.end(), 0);
    for (size_t f = 0; f < band_length; ++f)
    {
      const size_t base = static_cast<size_t>(band_start + static_cast<long>(f)) * npix;
      const float w = ww[f] / sumww;
      for (size_t iy = 0; iy < height; ++iy)
      {
        for (size_t ix = 0; ix < width; ++ix)
        {
          const size_t p = iy * width + ix;
          plane_img[p] += stokes_cube[base + p] * w;
          image_mask[p] = image_mask[p] || accept_mask_data[base + p];
        }
      }
    }
    for (size_t iy = 0; iy < height; ++iy)
    {
      for (size_t ix = 0; ix < width; ++ix)
      {
        const size_t p = iy * width + ix;
        any_nonzero = any_nonzero || (plane_img[p] != 0.f);
        any_masked = any_masked || image_mask[p];
      }
    }
  }

  if (!any_nonzero || !any_masked)
  {
    return {};
  }

  // SNSF: narrower bands get less spectral averaging, so more noise -- inflates selfnoise below.
  float bandwidth = 0.f;
  for (size_t f = 0; f < band_length; ++f)
  {
    bandwidth += channel_width(band_start + static_cast<long>(f));
  }
  const float snsf = full_bandwidth / bandwidth;

  // Pass 2: mean subtraction (optional) + abs_plane_img.
  std::vector<float> abs_plane_img(npix);
  if (subtract_mean && mean_image.has_value())
  {
    // Channel outermost (see pass 1); band_mean is a real extra buffer since the subtraction needs
    // the full cross-channel average first.
    const float *mean_data = mean_image->data();
    std::vector<float> band_mean(npix, 0.f);
    for (size_t f = 0; f < band_length; ++f)
    {
      const size_t base = static_cast<size_t>(band_start + static_cast<long>(f)) * npix;
      for (size_t iy = 0; iy < height; ++iy)
      {
        for (size_t ix = 0; ix < width; ++ix)
        {
          const size_t p = iy * width + ix;
          band_mean[p] += mean_data[base + p];
        }
      }
    }
    const float inv_n = 1.f / static_cast<float>(band_length);
    for (size_t iy = 0; iy < height; ++iy)
    {
      for (size_t ix = 0; ix < width; ++ix)
      {
        const size_t p = iy * width + ix;
        plane_img[p] -= band_mean[p] * inv_n;
        abs_plane_img[p] = std::abs(plane_img[p]);
      }
    }
  }
  else
  {
    for (size_t iy = 0; iy < height; ++iy)
    {
      for (size_t ix = 0; ix < width; ++ix)
      {
        const size_t p = iy * width + ix;
        abs_plane_img[p] = std::abs(plane_img[p]);
      }
    }
  }
  // abs_plane_img isn't needed again -- moved, not copied, into median().
  const float noise_globmad = median(std::move(abs_plane_img)) * 1.4826f;

  // Flat MAD estimator uses noise_globmad directly as a scalar below instead of an npix-sized constant array.
  const std::optional<std::vector<float>> noisemap0 =
      use_rms_noise_estimator ? std::optional(make_noise_map(plane_img, height, width, boxsize)) : std::nullopt;
  const float *noisemap0_data = noisemap0.has_value() ? noisemap0->data() : nullptr;

  // Pass 3: noisemap/excess_noise_factor/snr/abs_snr + image_mask &= valid + hist0.
  std::vector<float> noisemap(npix), excess_noise_factor(npix), snr(npix), abs_snr(npix);
  const float *selfnoise_data = selfnoise_offset.data();
  const float *dde_noise_data = dde_noise.data();
  std::vector<long> hist0;
  if (histogram_detail > 1)
  {
    hist0.assign(static_cast<size_t>(nbins), 0);
  }
  for (size_t iy = 0; iy < height; ++iy)
  {
    for (size_t ix = 0; ix < width; ++ix)
    {
      const size_t p = iy * width + ix;
      const float n0 = noisemap0_data ? noisemap0_data[p] : noise_globmad;
      const float variance = n0 * n0 + snsf * selfnoise_data[p] + ddenoise_scaling_factor * dde_noise_data[p];
      const float nm = std::sqrt(variance);
      noisemap[p] = nm;
      excess_noise_factor[p] = n0 != 0.f ? nm / n0 : 0.f;
      const bool valid = nm != 0.f && std::isfinite(nm);
      const float s = valid ? plane_img[p] / nm : 0.f;
      snr[p] = s;
      abs_snr[p] = std::abs(s);
      image_mask[p] = image_mask[p] && valid;

      if (!hist0.empty() && image_mask[p])
      {
        bin_into(hist0, s);
      }
    }
  }
  const float mednoise = median(noisemap);

  // Pass 4: SNR/flux/local-noise cutoffs + hist1/hist2 + minsig/maxsig + final_mask, matching
  // apply_rejection_cutoffs()'s order.
  const float *snr_cutoff_data = snr_cutoff.has_value() ? snr_cutoff->data() : nullptr;
  const float *flux_cutoff_data = flux_cutoff.has_value() ? flux_cutoff->data() : nullptr;
  const float inv_band_length = 1.f / static_cast<float>(band_length);
  const float noise_limit = mednoise * reject_noise_threshold;

  // Channel outermost (see pass 1); one shared pass over f for both cutoffs. Sums stay undivided --
  // /band_length is folded into the compare below.
  std::vector<float> snr_band_avg, flux_band_avg;
  if (snr_cutoff_data)
  {
    snr_band_avg.assign(npix, 0.f);
  }
  if (flux_cutoff_data)
  {
    flux_band_avg.assign(npix, 0.f);
  }
  if (snr_cutoff_data || flux_cutoff_data)
  {
    for (size_t f = 0; f < band_length; ++f)
    {
      const size_t base = static_cast<size_t>(band_start + static_cast<long>(f)) * npix;
      for (size_t iy = 0; iy < height; ++iy)
      {
        for (size_t ix = 0; ix < width; ++ix)
        {
          const size_t p = iy * width + ix;
          if (snr_cutoff_data)
          {
            snr_band_avg[p] += snr_cutoff_data[base + p];
          }
          if (flux_cutoff_data)
          {
            flux_band_avg[p] += flux_cutoff_data[base + p];
          }
        }
      }
    }
  }

  std::vector<long> hist1, hist2;
  if (histogram_detail > 1)
  {
    hist1.assign(static_cast<size_t>(nbins), 0);
  }
  if (histogram_detail > 0)
  {
    hist2.assign(static_cast<size_t>(nbins), 0);
  }

  bool any_mask_after_cutoffs = false;
  float minsig = std::numeric_limits<float>::infinity();
  float maxsig = -std::numeric_limits<float>::infinity();
  std::vector<uint8_t> final_mask(npix);

  for (size_t iy = 0; iy < height; ++iy)
  {
    for (size_t ix = 0; ix < width; ++ix)
    {
      const size_t p = iy * width + ix;
      bool m = image_mask[p];

      if (snr_cutoff_data)
      {
        m = m && (abs_snr[p] >= snr_band_avg[p] * inv_band_length);
      }
      if (!hist1.empty() && m)
      {
        bin_into(hist1, snr[p]);
      }

      if (flux_cutoff_data)
      {
        m = m && (std::abs(plane_img[p]) >= flux_band_avg[p] * inv_band_length);
      }
      if (!hist2.empty() && m)
      {
        bin_into(hist2, snr[p]);
      }

      if (reject_noise_threshold != 0.f)
      {
        m = m && (noisemap[p] <= noise_limit);
      }

      if (m)
      {
        any_mask_after_cutoffs = true;
        minsig = std::min(minsig, snr[p]);
        maxsig = std::max(maxsig, snr[p]);
      }
      final_mask[p] = (m ? abs_snr[p] : 0.f) >= threshold;
    }
  }

  if (!any_mask_after_cutoffs)
  {
    ProcessBandSliceResult result;
    result.processed = false;
    result.reached_cutoffs = true;
    result.mednoise = mednoise;
    result.noise_globmad = noise_globmad;
    result.hist0 = std::move(hist0);
    result.hist1 = std::move(hist1);
    result.hist2 = std::move(hist2);
    return result;
  }

  // Only reached if a pixel survived the cutoffs -- must NOT write back on the early return above.
  if (is_single_channel)
  {
    float *dst = stokes_cube + static_cast<size_t>(band_start) * npix;
    std::copy(snr.begin(), snr.end(), dst);
  }

  // xt::xarray only at the boundary -- ProcessBandSliceResult's field types are shared with process_plane_core().
  ProcessBandSliceResult result;
  result.processed = true;
  result.reached_cutoffs = true;
  result.snr = xt::zeros<float>({height, width});
  std::copy(snr.begin(), snr.end(), result.snr.begin());
  result.mask = xt::zeros<bool>({height, width});
  std::copy(final_mask.begin(), final_mask.end(), result.mask.begin());
  result.plane_img = xt::zeros<float>({height, width});
  std::copy(plane_img.begin(), plane_img.end(), result.plane_img.begin());
  result.noisemap = xt::zeros<float>({height, width});
  std::copy(noisemap.begin(), noisemap.end(), result.noisemap.begin());
  result.excess_noise_factor = xt::zeros<float>({height, width});
  std::copy(excess_noise_factor.begin(), excess_noise_factor.end(), result.excess_noise_factor.begin());
  result.mednoise = mednoise;
  result.noise_globmad = noise_globmad;
  result.minsig = minsig;
  result.maxsig = maxsig;
  result.hist0 = std::move(hist0);
  result.hist1 = std::move(hist1);
  result.hist2 = std::move(hist2);
  return result;
}

// One 4-connected island from label_islands(): bounding box (rows [r0,r1), cols [c0,c1)) plus a
// membership mask local to that bbox, in the same label order scipy.ndimage.label() would assign
// (order of first-encountered pixel during a row-major scan) -- load-bearing, since
// process_plane_core() appends one detection-array entry per island in this order.
struct Island
{
  long r0 = 0, r1 = 0, c0 = 0, c1 = 0;
  xt::xarray<bool> objmask;
};

/**
 * @brief 4-connected-component labeling of a boolean mask, with per-island bounding boxes.
 *
 * Port of `scipy.ndimage.label(mask)` (default structuring element -- orthogonal neighbors only,
 * no diagonals) followed by `scipy.ndimage.find_objects()`. Single row-major raster scan; on the
 * first unlabeled `true` pixel, flood-fills its connected component (iterative, stack-based) while
 * tracking the running bounding box. Islands are returned in scipy's own label-assignment order
 * (order of first-encountered pixel), which matters: process_plane_core() appends one detection per
 * island in this order.
 *
 * @param mask (H, W) boolean mask to label.
 * @return One Island per 4-connected component, in label order.
 */
inline std::vector<Island> label_islands(const xt::xarray<bool> &mask)
{
  const long h = static_cast<long>(mask.shape(0));
  const long w = static_cast<long>(mask.shape(1));
  xt::xarray<int> labels = xt::zeros<int>({static_cast<size_t>(h), static_cast<size_t>(w)});
  std::vector<Island> islands;

  std::vector<std::pair<long, long>> stack;
  for (long i = 0; i < h; ++i)
  {
    for (long j = 0; j < w; ++j)
    {
      if (!mask(i, j) || labels(i, j) != 0)
      {
        continue;
      }
      const int label_id = static_cast<int>(islands.size()) + 1;
      long r0 = i, r1 = i + 1, c0 = j, c1 = j + 1;
      labels(i, j) = label_id;
      stack.clear();
      stack.emplace_back(i, j);
      while (!stack.empty())
      {
        const auto [r, c] = stack.back();
        stack.pop_back();
        r0 = std::min(r0, r);
        r1 = std::max(r1, r + 1);
        c0 = std::min(c0, c);
        c1 = std::max(c1, c + 1);
        static constexpr long dr[4] = {-1, 1, 0, 0};
        static constexpr long dc[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; ++k)
        {
          const long nr = r + dr[k], nc = c + dc[k];
          if (nr < 0 || nr >= h || nc < 0 || nc >= w)
          {
            continue;
          }
          if (!mask(nr, nc) || labels(nr, nc) != 0)
          {
            continue;
          }
          labels(nr, nc) = label_id;
          stack.emplace_back(nr, nc);
        }
      }
      xt::xarray<bool> objmask = xt::equal(xt::view(labels, xt::range(r0, r1), xt::range(c0, c1)), label_id);
      islands.push_back(Island{r0, r1, c0, c1, std::move(objmask)});
    }
  }
  return islands;
}

/**
 * @brief Expected median of the max of `n` iid standard-normal samples.
 *
 * Port of breifast.py's max_to_std_ratio(): builds F(x) = 1 - (1 - Phi(x))^n over a 1000-point grid
 * spanning [-10, 10] (Phi = standard normal CDF via std::erf), then linearly interpolates to find x
 * such that F(x) = 0.5 -- matching numpy.interp's clamp-at-the-ends behavior for out-of-range query
 * points against the (monotonically increasing) F grid. Memoized by `n`, since only a handful of
 * distinct postage-stamp sizes recur across islands/bands in one run.
 *
 * @param n Sample count (postage-stamp pixel count).
 * @return Expected median of the max of `n` iid N(0,1) samples.
 */
inline float max_to_std_ratio(long n)
{
  static std::mutex mutex;
  static std::unordered_map<long, float> cache;

  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(n);
    if (it != cache.end())
    {
      return it->second;
    }
  }

  constexpr int npoints = 1000;
  constexpr double xmin = -10.0, xmax = 10.0;
  double prev_x = xmin, prev_F = 0.0;
  double result = 0.0;
  bool found = false;
  for (int i = 0; i < npoints; ++i)
  {
    const double x = xmin + (xmax - xmin) * static_cast<double>(i) / static_cast<double>(npoints - 1);
    const double phi = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    const double F = 1.0 - std::pow(1.0 - phi, static_cast<double>(n));
    if (i > 0 && !found && prev_F <= 0.5 && F >= 0.5)
    {
      const double t = (F != prev_F) ? (0.5 - prev_F) / (F - prev_F) : 0.0;
      result = prev_x + t * (x - prev_x);
      found = true;
    }
    prev_x = x;
    prev_F = F;
  }
  if (!found)
  {
    result = (prev_F < 0.5) ? xmax : xmin;
  }

  const float ratio = static_cast<float>(std::abs(result));
  std::lock_guard<std::mutex> lock(mutex);
  cache.emplace(n, ratio);
  return ratio;
}

/**
 * @brief Centred 2D Gaussian kernel of odd size (2*half_size+1)^2, normalized to sum to 1.
 *
 * Specialized port of create_multidimensional_gaussian_kernel(..., centred=True) restricted to 2
 * dimensions -- the only case this codebase ever calls it with (breifast-tron's cubes.py is fully
 * N-dimensional/general; every call site in process_plane() uses exactly 2 axes and centred=True).
 * For each pixel offset (dr, dc) from the kernel's center, optionally rotates (dr, dc) by
 * `rotate_radians` (counter-clockwise) before applying the per-axis Gaussian falloff; a zero sigma
 * on an axis contributes nothing to that axis's exponent (a delta-like ridge along the other axis).
 *
 * @param sigma_row Gaussian sigma along the row (first) axis, in pixels.
 * @param sigma_col Gaussian sigma along the column (second) axis, in pixels.
 * @param half_size Kernel half-size; full kernel is (2*half_size+1) x (2*half_size+1).
 * @param rotate_radians Rotation angle (radians, counter-clockwise) applied to coordinates before
 *        evaluating the per-axis Gaussian.
 * @return (2*half_size+1, 2*half_size+1) kernel, normalized to sum to 1.
 */
inline xt::xarray<double> create_multidimensional_gaussian_kernel(double sigma_row, double sigma_col, long half_size,
                                                                  double rotate_radians)
{
  const long size = 2 * half_size + 1;
  xt::xarray<double> kernel = xt::zeros<double>({static_cast<size_t>(size), static_cast<size_t>(size)});
  const double cos_r = std::cos(rotate_radians);
  const double sin_r = std::sin(rotate_radians);

  double total = 0.0;
  for (long i = 0; i < size; ++i)
  {
    const double dr = static_cast<double>(i - half_size);
    for (long j = 0; j < size; ++j)
    {
      const double dc = static_cast<double>(j - half_size);
      const double rr = cos_r * dr - sin_r * dc;
      const double rc = sin_r * dr + cos_r * dc;
      const double term_row = sigma_row != 0.0 ? (rr * rr) / (2.0 * sigma_row * sigma_row) : 0.0;
      const double term_col = sigma_col != 0.0 ? (rc * rc) / (2.0 * sigma_col * sigma_col) : 0.0;
      const double value = std::exp(-(term_row + term_col));
      kernel(i, j) = value;
      total += value;
    }
  }
  if (total != 0.0)
  {
    kernel /= total;
  }
  return kernel;
}

/**
 * @brief Zero-pads `plane_img` by `pad` pixels on every side.
 *
 * Built once per band and shared across all of that band's islands.
 *
 * @param plane_img (H, W) image to pad.
 * @param pad Padding amount, in pixels, on every side.
 * @return (H + 2*pad, W + 2*pad) zero-padded image.
 */
inline xt::xarray<float> pad_plane_img(const xt::xarray<float> &plane_img, long pad)
{
  const long h = static_cast<long>(plane_img.shape(0));
  const long w = static_cast<long>(plane_img.shape(1));
  xt::xarray<float> padded = xt::zeros<float>({static_cast<size_t>(h + 2 * pad), static_cast<size_t>(w + 2 * pad)});
  xt::view(padded, xt::range(pad, pad + h), xt::range(pad, pad + w)) = plane_img;
  return padded;
}

// Per-island stats that don't need a postage stamp -- see find_island_peak().
struct IslandPeak
{
  long x = 0, y = 0;
  float maxsig = 0.f;
  float flux = 0.f;
  float boost = 0.f;
  long size = 0;
  float local_noise = 0.f;
  float selfnoise_value = 0.f;
  float excess_noise_factor_value = 0.f;
  float sigma_gmad = 0.f;
};

/**
 * @brief Finds an island's peak-|snr| pixel and every per-island stat that doesn't need a postage
 * stamp.
 *
 * Tie-breaking matches numpy exactly: the first pixel, in row-major traversal of the island's own
 * bounding box restricted to its `objmask`, attaining max(|snr|).
 *
 * @param island Island (bbox + membership mask) from label_islands().
 * @param snr Full-plane per-pixel SNR image.
 * @param plane_img Full-plane flux image.
 * @param noisemap Full-plane per-pixel noise map.
 * @param noise_globmad Global MAD noise estimate for this band.
 * @param baseline_flux Optional (H, W) baseline flux image; `boost` is 0 if absent.
 * @param selfnoise_raw Full-plane raw (unscaled) selfnoise image.
 * @param excess_noise_factor Full-plane per-pixel excess noise factor image.
 * @return See IslandPeak.
 */
inline IslandPeak find_island_peak(const Island &island, const xt::xarray<float> &snr,
                                   const xt::xarray<float> &plane_img, const xt::xarray<float> &noisemap,
                                   float noise_globmad, const std::optional<xt::xarray<float>> &baseline_flux,
                                   const xt::xarray<float> &selfnoise_raw, const xt::xarray<float> &excess_noise_factor)
{
  IslandPeak peak;
  float maxabs = -1.f;
  long peak_row = island.r0, peak_col = island.c0;
  long size = 0;
  float local_noise = -std::numeric_limits<float>::infinity();

  const long rows = island.r1 - island.r0;
  const long cols = island.c1 - island.c0;
  for (long i = 0; i < rows; ++i)
  {
    for (long j = 0; j < cols; ++j)
    {
      if (!island.objmask(i, j))
      {
        continue;
      }
      const long r = island.r0 + i, c = island.c0 + j;
      ++size;
      local_noise = std::max(local_noise, noisemap(r, c));
      const float abssig = std::abs(snr(r, c));
      if (abssig > maxabs)
      {
        maxabs = abssig;
        peak_row = r;
        peak_col = c;
      }
    }
  }

  peak.x = peak_col;
  peak.y = peak_row;
  peak.maxsig = maxabs;
  peak.size = size;
  peak.local_noise = local_noise;
  peak.flux = plane_img(peak_row, peak_col);
  peak.selfnoise_value = selfnoise_raw(peak_row, peak_col);
  peak.excess_noise_factor_value = excess_noise_factor(peak_row, peak_col);
  // Not guarded, matching breifast.py's unguarded `sig_gmad = flmax / noise_globmad`.
  peak.sigma_gmad = peak.flux / noise_globmad;

  peak.boost = 0.f;
  if (baseline_flux.has_value())
  {
    float best_ratio = 0.f;
    float best_abs = -1.f;
    for (long i = 0; i < rows; ++i)
    {
      for (long j = 0; j < cols; ++j)
      {
        if (!island.objmask(i, j))
        {
          continue;
        }
        const long r = island.r0 + i, c = island.c0 + j;
        // Not guarded, matching breifast.py's unguarded `flmax / baseline_flux[obj_slice][objmask]`.
        const float ratio = peak.flux / (*baseline_flux)(r, c);
        const float aratio = std::abs(ratio);
        if (aratio > best_abs)
        {
          best_abs = aratio;
          best_ratio = ratio;
        }
      }
    }
    peak.boost = best_ratio;
  }

  return peak;
}

// Adjusted-noise-based sigma estimates for one island -- see compute_adjusted_noise_stats().
struct AdjustedNoiseStats
{
  float sig_ext = 0.f;
  float sig_rms = 0.f;
  float sig_lmad = 0.f;
  float adjnoise = 0.f;
};

/**
 * @brief Computes the adjusted-noise-based sigma estimates for one island from its postage stamp,
 * and overwrites `snr`'s island pixels in place with sigma_ext.
 *
 * Matches `snr[obj_slice][objmask] = sig_ext` (breifast.py:992) -- this is what later gets written
 * back into stokes_cube for single-channel bands (see process_plane_core()'s control flow).
 *
 * @param island Island (bbox + membership mask) whose pixels get overwritten in `snr`.
 * @param snr Full-plane per-pixel SNR image, mutated in place at this island's pixels.
 * @param padimg Zero-padded plane_img (see pad_plane_img()), padded by `pad` on every side.
 * @param peak_x,peak_y Peak pixel position (unpadded coordinates), from find_island_peak().
 * @param pad Padding amount used to build `padimg`.
 * @param boxsize Side length of the RMS noise estimator's sliding box; `r = boxsize/2` is the
 *        adjusted-noise postage stamp's half-size.
 * @param inner_boxsize Half-size of the inner box zeroed out to exclude the source itself.
 * @param flux Peak flux value (flmax in the Python original).
 * @return See AdjustedNoiseStats.
 */
inline AdjustedNoiseStats compute_adjusted_noise_stats(const Island &island, xt::xarray<float> &snr,
                                                       const xt::xarray<float> &padimg, long peak_x, long peak_y,
                                                       long pad, long boxsize, long inner_boxsize, float flux)
{
  const long r = boxsize / 2;
  const long r1 = inner_boxsize;
  const long px = peak_x + pad, py = peak_y + pad;

  xt::xarray<float> subimg =
      xt::xarray<float>(xt::view(padimg, xt::range(py - r, py + r + 1), xt::range(px - r, px + r + 1)));
  xt::view(subimg, xt::range(r - r1, r + r1 + 1), xt::range(r - r1, r + r1 + 1)) = 0.f;

  const float extreme = std::max(std::abs(xt::amax(subimg)()), std::abs(xt::amin(subimg)()));
  AdjustedNoiseStats stats;
  stats.adjnoise = extreme / max_to_std_ratio(static_cast<long>(subimg.size()));
  // Guarded exactly like breifast.py:991's `sig_ext = flmax / adjnoise if adjnoise else 0`.
  stats.sig_ext = stats.adjnoise != 0.f ? flux / stats.adjnoise : 0.f;

  // "donut": the postage stamp excluding exact-zero pixels (the zeroed inner box, plus any
  // edge-padding zeros) -- matches Python's `ma.masked_array(subimg, subimg==0)`.
  double sum = 0.0, sum_sq = 0.0;
  std::vector<float> abs_values;
  abs_values.reserve(subimg.size());
  for (auto v : subimg)
  {
    if (v == 0.f)
    {
      continue;
    }
    sum += v;
    sum_sq += static_cast<double>(v) * v;
    abs_values.push_back(std::abs(v));
  }
  const long count = static_cast<long>(abs_values.size());

  if (count > 0)
  {
    const double mean = sum / count;
    const double variance = std::max(0.0, sum_sq / count - mean * mean);
    const float rms = static_cast<float>(std::sqrt(variance));
    // Guarded exactly like breifast.py:995's `sig_rms = flmax / rms if rms else 0`.
    stats.sig_rms = rms != 0.f ? flux / rms : 0.f;

    std::sort(abs_values.begin(), abs_values.end());
    const float median_abs =
        (count % 2 == 1)
            ? abs_values[static_cast<size_t>(count / 2)]
            : 0.5f * (abs_values[static_cast<size_t>(count / 2 - 1)] + abs_values[static_cast<size_t>(count / 2)]);
    // NOT guarded, matching breifast.py:996's unguarded division (produces +-inf if median_abs==0,
    // same as the unguarded numpy division it ports).
    stats.sig_lmad = flux / (median_abs * 1.4826f);
  }
  else
  {
    // Postage stamp fully zero after masking: Python's donut.std()/ma.median() both degrade to the
    // numpy.ma.masked sentinel here, which has no clean numeric C++ equivalent -- rather than
    // propagate NaN/inf from a genuinely degenerate case, fall back to 0 for both (a deliberate,
    // pragmatic deviation from a literal transcription; flagged for follow-up if real fixtures ever
    // exercise this path).
    stats.sig_rms = 0.f;
    stats.sig_lmad = 0.f;
  }

  const long rows = island.r1 - island.r0;
  const long cols = island.c1 - island.c0;
  for (long i = 0; i < rows; ++i)
  {
    for (long j = 0; j < cols; ++j)
    {
      if (island.objmask(i, j))
      {
        snr(island.r0 + i, island.c0 + j) = stats.sig_ext;
      }
    }
  }

  return stats;
}

/**
 * @brief Beam-fit reduced chi-square for one island, given a (lazily-built-once-per-band, cached)
 * Gaussian beam kernel.
 *
 * Reuses `adjnoise` from compute_adjusted_noise_stats()'s (differently-sized, source-zeroed)
 * postage stamp -- matches breifast.py:1017's reuse across the two different stamp sizes. This
 * stamp is NOT source-zeroed (unlike the adjusted-noise one), matching breifast.py:1014.
 *
 * @param beam_kernel (2*beam_halfsize+1, 2*beam_halfsize+1) beam kernel, normalized to sum to 1.
 * @param beam_halfsize Half-size of the beam-fit postage stamp.
 * @param padimg Zero-padded plane_img (see pad_plane_img()).
 * @param peak_x,peak_y Peak pixel position (unpadded coordinates).
 * @param pad Padding amount used to build `padimg`.
 * @param adjnoise Adjusted noise estimate from compute_adjusted_noise_stats(), reused here.
 * @return Reduced chi-square of the scaled-kernel fit against the postage stamp.
 */
inline float compute_beam_chi2(const xt::xarray<double> &beam_kernel, long beam_halfsize,
                               const xt::xarray<float> &padimg, long peak_x, long peak_y, long pad, float adjnoise)
{
  const long px = peak_x + pad, py = peak_y + pad;
  const xt::xarray<double> subimg =
      xt::xarray<double>(xt::view(padimg, xt::range(py - beam_halfsize, py + beam_halfsize + 1),
                                  xt::range(px - beam_halfsize, px + beam_halfsize + 1)));

  const double subimg_sum = xt::sum(subimg)();
  const double kernel_sum = xt::sum(beam_kernel)();
  const xt::xarray<double> fit = beam_kernel * (subimg_sum / kernel_sum);

  double chi2_sum = 0.0;
  long count = 0;
  auto it_s = subimg.begin();
  auto it_f = fit.begin();
  for (; it_s != subimg.end(); ++it_s, ++it_f)
  {
    if (*it_s == 0.0)
    {
      continue;
    }
    const double diff = *it_s - *it_f;
    chi2_sum += diff * diff;
    ++count;
  }

  // Not guarded, matching breifast.py:1017's unguarded division (produces +-inf/NaN if adjnoise==0
  // or the postage stamp is fully zero, same as the unguarded numpy division it ports).
  const double denom = static_cast<double>(adjnoise) * static_cast<double>(adjnoise) * static_cast<double>(count);
  return static_cast<float>(chi2_sum / denom);
}

// Per-pixel spectral variance/skew/kurtosis over nonzero channels -- see
// compute_spectral_variance_stats().
struct SpectralVarianceStats
{
  xt::xarray<double> spvar;
  xt::xarray<double> spskew;
  xt::xarray<double> spkurt;
  long counts = 0;
  std::optional<xt::xarray<float>> timevar;
};

/**
 * @brief Per-pixel spectral variance/skew/kurtosis over nonzero channels of `stokes_cube`.
 *
 * Computed once per (interval, stokes) call, after the whole reversed band loop has finished (so
 * `stokes_cube` is already SNR-and-island-adjusted for every single-channel band by this point).
 * Sums (not divided by `counts`) are returned, matching what breifast.py's process_plane() passes
 * onward via `variance_stats` for the caller to divide/write to storage itself -- zarr writes and
 * the division stay Python-side (see process_plane_core()'s docstring for what's out of scope).
 *
 * @param stokes_cube (FREQ, H, W) cube for this Stokes/interval, post band-processing.
 * @param nonzero_channels (FREQ,) which channels to include.
 * @param save_stat_cubes Whether to additionally return the full per-channel squared cube.
 * @return See SpectralVarianceStats.
 */
inline SpectralVarianceStats compute_spectral_variance_stats(const xt::xarray<float> &stokes_cube,
                                                             const xt::xarray<bool> &nonzero_channels,
                                                             bool save_stat_cubes)
{
  const long nfreq = static_cast<long>(stokes_cube.shape(0));
  const size_t h = stokes_cube.shape(1);
  const size_t w = stokes_cube.shape(2);

  SpectralVarianceStats result;
  result.spvar = xt::zeros<double>({h, w});
  result.spskew = xt::zeros<double>({h, w});
  result.spkurt = xt::zeros<double>({h, w});

  for (long f = 0; f < nfreq; ++f)
  {
    if (!nonzero_channels(f))
    {
      continue;
    }
    ++result.counts;
    const xt::xarray<double> channel = xt::view(stokes_cube, f, xt::all(), xt::all());
    const xt::xarray<double> sq = channel * channel;
    result.spvar += sq;
    result.spskew += sq * channel;
    result.spkurt += sq * sq;
  }

  if (save_stat_cubes)
  {
    result.timevar = xt::xarray<float>(stokes_cube * stokes_cube);
  }

  return result;
}

// One band's channel range + label, matching breifast.py's `(slice, band_id)` tuples in
// `band_slices`. Forward order (as constructed in Python); process_plane_core() reverses
// internally.
struct BandSlice
{
  long start = 0, stop = 0;
  std::string label;
};

// Read-only, per-plane-call settings -- everything that doesn't change across bands/islands within
// one (interval, stokes) call. Kept separate from the array/payload parameters of
// process_plane_core() (which stay flat, matching process_band_slice_core()'s existing style)
// because struct members that are array references have awkward default-construction/reassignment
// semantics -- see this header's own design notes in the implementation plan for the reasoning.
// Also the natural seam for a later batching step: many (interval, stokes) items processed by one
// future call could share a single `const PlaneConfig&`.
struct PlaneConfig
{
  std::vector<BandSlice> band_slices;
  bool subtract_mean = false;
  bool use_rms_noise_estimator = true;
  int boxsize = 0;
  int inner_boxsize = 10;
  float ddenoise_scaling_factor = 1.f;
  float full_bandwidth = 0.f;
  float reject_noise_threshold = 0.f;
  int histogram_detail = 0;
  int max_sigma_bin = 200;
  float threshold = 0.f;
  bool save_stat_cubes = false;
  bool beam_size_enabled = false;
  std::optional<xt::xarray<double>> fixed_beam_kernel; // set iff Python's beam_kernel is not None
  long fixed_beam_halfsize = 0;                        // only meaningful if fixed_beam_kernel is set
};

// One detection per island, across all processed bands -- parallel arrays instead of a table
// object, since DetectionTable (astropy QTable-backed) is Python-metadata-driven and stays
// Python-side; the pybind11 wrapper hands these back for Python to assemble into a DetectionTable
// exactly as it does today.
struct DetectionArrays
{
  std::vector<long> x, y, size;
  std::vector<std::string> band_label;
  std::vector<float> peak_flux, sigma, sigma_ext, sigma_rms, sigma_lmad, sigma_gmad;
  std::vector<float> boost, local_noise, ln_x_curmed, selfnoise_value, excess_noise_factor_value, beamchi2;
};

// Result of process_plane_core(). Mirrors the 8-element tuple (plus DetectionTable expanded into
// raw arrays) that breifast.py's process_plane() returns.
struct ProcessPlaneResult
{
  bool any_data = false;
  DetectionArrays detections;
  std::optional<float> minsig, maxsig;
  std::vector<long> hist0, hist1, hist2;
  std::vector<double> mednoises;                       // indexed by reversed-loop position (iband)
  std::optional<SpectralVarianceStats> variance_stats; // present iff any_data && nfreq > 1
};

/**
 * @brief Computes island detections and per-plane spectral variance stats for one (interval,
 * stokes) plane, across all of its band slices.
 *
 * Port of the rest of process_plane() not already covered by process_band_slice_core(): island
 * detection, per-island stats, optional beam fitting, and spectral variance. Calls
 * process_band_slice_core() once per band, unmodified, reusing its own internal (pre-island) SNR
 * write-back to `stokes_cube` for single-channel bands -- but overwrites that same channel a
 * second time, after island adjustment, whenever islands are found (see the control-flow comments
 * in the implementation below).
 *
 * Explicitly out of scope, staying Python-side: DetectionTable construction itself, FITS debug
 * dumps, zarr writes for variance_stats, and Stokes-P plane formation (a cheap, cross-Stokes op
 * done once before any per-band processing) -- this function always receives an already-resolved
 * single Stokes plane, same as process_band_slice_core() today.
 *
 * @param stokes_cube (FREQ, H, W) flux cube for this Stokes/interval. Mutated in place (see above).
 * @param weights (FREQ,) per-channel weights for this Stokes/interval.
 * @param accept_mask (FREQ, H, W) per-pixel acceptance mask for this Stokes.
 * @param channel_width (FREQ,) full per-channel bandwidths.
 * @param nonzero_channels (FREQ,) which channels have any data -- serves the top-level any-data
 *        gate, the per-band skip check, and the spectral-variance nonzero mask.
 * @param mean_image (FREQ, H, W) mean image for this Stokes, or absent if `subtract_mean` is false.
 * @param selfnoise_offset (H, W) per-pixel selfnoise variance term (scaled, used in noise combination).
 * @param selfnoise_raw (H, W) raw (unscaled) selfnoise, sampled at each island's peak for output.
 * @param dde_noise (H, W) per-pixel DDE-calibration noise variance term.
 * @param flux_cutoff (FREQ, H, W) per-pixel flux cutoff image, or absent to skip.
 * @param snr_cutoff (FREQ, H, W) per-pixel SNR cutoff image, or absent to skip.
 * @param baseline_flux Optional (H, W) baseline flux image for the `boost` detection stat.
 * @param bmaj_per_channel,bmin_per_channel,bpa_per_channel Optional (FREQ,) beam-shape arrays for
 *        this interval, required whenever `config.beam_size_enabled` and no `fixed_beam_kernel`.
 * @param config Read-only per-plane-call settings; see PlaneConfig.
 * @return See ProcessPlaneResult.
 */
inline ProcessPlaneResult process_plane_core(
    xt::xarray<float> &stokes_cube, const xt::xarray<float> &weights, const xt::xarray<bool> &accept_mask,
    const xt::xarray<float> &channel_width, const xt::xarray<bool> &nonzero_channels,
    const std::optional<xt::xarray<float>> &mean_image, const xt::xarray<float> &selfnoise_offset,
    const xt::xarray<float> &selfnoise_raw, const xt::xarray<float> &dde_noise,
    const std::optional<xt::xarray<float>> &flux_cutoff, const std::optional<xt::xarray<float>> &snr_cutoff,
    const std::optional<xt::xarray<float>> &baseline_flux, const std::optional<xt::xarray<float>> &bmaj_per_channel,
    const std::optional<xt::xarray<float>> &bmin_per_channel, const std::optional<xt::xarray<float>> &bpa_per_channel,
    const PlaneConfig &config)
{
  ProcessPlaneResult result;
  result.any_data = xt::any(nonzero_channels);
  if (!result.any_data)
  {
    return result;
  }

  const long num_bands = static_cast<long>(config.band_slices.size());
  std::vector<double> mednoises(static_cast<size_t>(num_bands), 0.0);
  std::vector<long> hist0, hist1, hist2;
  std::optional<float> minsig, maxsig;
  DetectionArrays detections;

  // Reverse order matches breifast.py:659's `band_slices[::-1]`: band_slices is
  // [per-channel single bands] + [aggregated multi-channel bands] in forward order, so reversed
  // iteration processes aggregated bands FIRST -- they read raw flux from multiple channels via a
  // weighted mean -- before any single-channel band's processing overwrites those same channels'
  // data in stokes_cube with SNR values.
  for (long iband = 0; iband < num_bands; ++iband)
  {
    const BandSlice &band = config.band_slices[static_cast<size_t>(num_bands - 1 - iband)];

    if (!xt::any(xt::view(nonzero_channels, xt::range(band.start, band.stop))))
    {
      continue;
    }

    const bool is_single_channel = (band.stop - band.start) == 1;
    ProcessBandSliceResult band_result = process_band_slice_core(
        stokes_cube, weights, accept_mask, channel_width, band.start, band.stop, is_single_channel,
        config.subtract_mean, mean_image, config.use_rms_noise_estimator, config.boxsize, selfnoise_offset, dde_noise,
        config.ddenoise_scaling_factor, config.full_bandwidth, flux_cutoff, snr_cutoff, config.reject_noise_threshold,
        config.histogram_detail, config.max_sigma_bin, config.threshold);

    // Correct in all three of process_band_slice_core()'s outcomes (very-early-return default 0,
    // post-cutoff-empty real mednoise, full success), matching breifast.py:770.
    mednoises[static_cast<size_t>(iband)] = band_result.mednoise;

    // NOTE: overwrites (not accumulates) hist0/1/2 on every band that reaches the cutoffs --
    // faithfully replicating breifast.py:781's `hist0 = hist1 = hist2 = None` reset at the top of
    // every band iteration, which makes the Python `else: hist0 += ...` accumulation branch dead
    // code. This is a known quirk of the upstream pipeline, replicated here by design (not fixed),
    // per the fidelity goal for this port -- worth a conversation with that codebase's maintainers.
    if (band_result.reached_cutoffs)
    {
      hist0 = band_result.hist0;
      hist1 = band_result.hist1;
      hist2 = band_result.hist2;
    }

    if (!band_result.processed)
    {
      continue;
    }

    minsig = minsig.has_value() ? std::min(*minsig, band_result.minsig) : band_result.minsig;
    maxsig = maxsig.has_value() ? std::max(*maxsig, band_result.maxsig) : band_result.maxsig;

    if (xt::any(band_result.mask))
    {
      const std::vector<Island> islands = label_islands(band_result.mask);
      const long r = config.boxsize / 2;
      long pad;
      long bk_halfsize = 0;
      if (config.beam_size_enabled)
      {
        // NOTE: bk_halfsize uses an UNWEIGHTED mean of bmaj (breifast.py:941), while the kernel's
        // own sigma below uses a weight-weighted mean of the same bmaj (breifast.py:1009) -- a real
        // inconsistency in the upstream Python, replicated faithfully here per the fidelity goal for
        // this port (not fixed); worth a conversation with that codebase's maintainers.
        bk_halfsize = config.fixed_beam_kernel.has_value()
                          ? config.fixed_beam_halfsize
                          : static_cast<long>(std::round(
                                xt::mean(xt::view(*bmaj_per_channel, xt::range(band.start, band.stop)))() * 3.0)) +
                                1;
        pad = std::max(bk_halfsize, r + 1);
      }
      else
      {
        pad = r + 1;
      }
      const xt::xarray<float> padimg = pad_plane_img(band_result.plane_img, pad);

      std::optional<xt::xarray<double>> band_beam_kernel = config.fixed_beam_kernel;
      const xt::xarray<float> ww = xt::view(weights, xt::range(band.start, band.stop));
      const float sumww = xt::sum(ww)();

      for (const Island &island : islands)
      {
        const IslandPeak peak =
            find_island_peak(island, band_result.snr, band_result.plane_img, band_result.noisemap,
                             band_result.noise_globmad, baseline_flux, selfnoise_raw, band_result.excess_noise_factor);
        const AdjustedNoiseStats adj = compute_adjusted_noise_stats(
            island, band_result.snr, padimg, peak.x, peak.y, pad, config.boxsize, config.inner_boxsize, peak.flux);

        float beamchi2 = 0.f;
        if (config.beam_size_enabled)
        {
          // Built lazily on the first island of this band, then reused for the rest -- matching
          // breifast.py:1005-1007's `bk = None` / `elif bk is None`.
          if (!band_beam_kernel.has_value())
          {
            const float sigma_row =
                xt::sum(xt::view(*bmaj_per_channel, xt::range(band.start, band.stop)) * ww)() / sumww;
            const float sigma_col =
                xt::sum(xt::view(*bmin_per_channel, xt::range(band.start, band.stop)) * ww)() / sumww;
            const float rotate = xt::sum(xt::view(*bpa_per_channel, xt::range(band.start, band.stop)) * ww)() / sumww;
            band_beam_kernel = create_multidimensional_gaussian_kernel(sigma_row, sigma_col, bk_halfsize, rotate);
          }
          beamchi2 = compute_beam_chi2(*band_beam_kernel, bk_halfsize, padimg, peak.x, peak.y, pad, adj.adjnoise);
        }

        detections.x.push_back(peak.x);
        detections.y.push_back(peak.y);
        detections.band_label.push_back(band.label);
        detections.peak_flux.push_back(peak.flux);
        detections.sigma.push_back(peak.maxsig);
        detections.sigma_ext.push_back(adj.sig_ext);
        detections.sigma_rms.push_back(adj.sig_rms);
        detections.sigma_lmad.push_back(adj.sig_lmad);
        detections.sigma_gmad.push_back(peak.sigma_gmad);
        detections.boost.push_back(peak.boost);
        detections.size.push_back(peak.size);
        detections.local_noise.push_back(peak.local_noise);
        // NOT guarded, matching breifast.py:1024's unguarded division.
        detections.ln_x_curmed.push_back(peak.local_noise / band_result.mednoise);
        detections.selfnoise_value.push_back(peak.selfnoise_value);
        detections.excess_noise_factor_value.push_back(peak.excess_noise_factor_value);
        detections.beamchi2.push_back(beamchi2);
      }
    }

    // Second write-back, after island adjustment (breifast.py:1027-1028's `finally: plane_img[:] =
    // snr`, which runs after the island loop above already overwrote band_result.snr's island
    // pixels with sig_ext via compute_adjusted_noise_stats()). process_band_slice_core() already
    // wrote its own pre-island snr into stokes_cube internally; this second write only actually
    // changes anything when islands were found above.
    if (is_single_channel)
    {
      xt::view(stokes_cube, band.start, xt::all(), xt::all()) = band_result.snr;
    }
  }

  result.detections = std::move(detections);
  result.minsig = minsig;
  result.maxsig = maxsig;

  // Computed once per plane, after the whole reversed band loop -- so stokes_cube is already
  // SNR-and-island-adjusted for every single-channel band by this point. Matches breifast.py:995's
  // `if nfreq > 1`, using the WHOLE cube's channel count (not just this call's band_slices).
  // Preserved regardless of whether any band fully succeeded, matching breifast.py's "nothing at
  // all" return tuple, which still carries variance_stats through unchanged.
  if (stokes_cube.shape(0) > 1)
  {
    result.variance_stats = compute_spectral_variance_stats(stokes_cube, nonzero_channels, config.save_stat_cubes);
  }

  // Final collapse: matches breifast.py's `if minsig is None: return (..., None, None, None, None,
  // ...)` branch, which discards hist0/1/2 and mednoises entirely if no band ever fully succeeded --
  // even if an earlier band's reached_cutoffs/mednoises[iband] got populated along the way.
  if (minsig.has_value())
  {
    result.hist0 = std::move(hist0);
    result.hist1 = std::move(hist1);
    result.hist2 = std::move(hist2);
    result.mednoises = std::move(mednoises);
  }

  return result;
}

// One (interval, stokes) item within a process_plane_batch_core() call. Python-independent --
// shared by process_plane_batch()'s pybind11 wrapper (breifast_helpers.cpp, which additionally
// keeps a parallel xt::pyarray to write the mutated stokes_cube back into) and bench.cpp, so both
// exercise the exact same dispatch code below.
struct PlaneBatchItem
{
  size_t group_index = 0;
  xt::xarray<float> stokes_cube;
  xt::xarray<float> weights;
  xt::xarray<bool> nonzero_channels;
  std::optional<xt::xarray<float>> bmaj, bmin, bpa;
};

// Arrays shared by every item for one Stokes label within a process_plane_batch_core() call --
// accept_mask, mean_image, selfnoise, dde_noise, and the flux/snr cutoff images don't vary per
// interval, so callers build these once per Stokes label, not once per item.
struct PlaneBatchGroup
{
  xt::xarray<bool> accept_mask;
  std::optional<xt::xarray<float>> mean_image;
  xt::xarray<float> selfnoise_offset;
  xt::xarray<float> selfnoise_raw;
  xt::xarray<float> dde_noise;
  std::optional<xt::xarray<float>> flux_cutoff;
  std::optional<xt::xarray<float>> snr_cutoff;
};

struct PlaneBatchItemResult
{
  bool has_error = false;
  std::string error_message;
  ProcessPlaneResult result;
};

/**
 * @brief Runs process_plane_core() for every item in `items`, fanned out across a
 * tbb::parallel_for (optionally capped to `num_threads` via tbb::global_control), catching
 * exceptions per item so one bad plane doesn't abort the whole batch.
 *
 * This *is* the parallel-dispatch logic behind process_plane_batch() (the pybind11 entry point in
 * breifast_helpers.cpp) -- deliberately Python-independent so bench.cpp calls this same function
 * rather than reimplementing its own parallel loop. The pybind11 wrapper's job is only Python <->
 * C++ marshaling and GIL release/reacquire around this call; it must not duplicate any
 * dispatch/threading/error-handling logic itself, on pain of the benchmark (or any other caller)
 * silently drifting from what the real pipeline actually does.
 *
 * @param items Items to process; mutated in place (each item's stokes_cube gets process_plane_core()'s
 *        in-place SNR write-back).
 * @param groups Per-Stokes-label shared arrays, indexed by each item's group_index.
 * @param channel_width (FREQ,) full per-channel bandwidths, shared by every item.
 * @param baseline_flux Optional (H, W) baseline flux image, shared by every item.
 * @param config Read-only per-plane-call settings, shared by every item.
 * @param num_threads Caps the TBB thread pool for this call; <= 0 means TBB's own default (all
 *        hardware threads).
 * @return One result per item, in the same order as `items`.
 */
inline std::vector<PlaneBatchItemResult> process_plane_batch_core(std::vector<PlaneBatchItem> &items,
                                                                  const std::vector<PlaneBatchGroup> &groups,
                                                                  const xt::xarray<float> &channel_width,
                                                                  const std::optional<xt::xarray<float>> &baseline_flux,
                                                                  const PlaneConfig &config, int num_threads)
{
  std::vector<PlaneBatchItemResult> results(items.size());

  std::unique_ptr<tbb::global_control> concurrency_limit;
  if (num_threads > 0)
  {
    concurrency_limit = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism,
                                                              static_cast<size_t>(num_threads));
  }
  std::cout << items.size() << std::endl;
  tbb::parallel_for(static_cast<size_t>(0), items.size(),
                    [&](size_t i)
                    {
                      PlaneBatchItem &item = items[i];
                      const PlaneBatchGroup &group = groups[item.group_index];
                      PlaneBatchItemResult &out = results[i];
                      try
                      {
                        out.result = process_plane_core(
                            item.stokes_cube, item.weights, group.accept_mask, channel_width, item.nonzero_channels,
                            group.mean_image, group.selfnoise_offset, group.selfnoise_raw, group.dde_noise,
                            group.flux_cutoff, group.snr_cutoff, baseline_flux, item.bmaj, item.bmin, item.bpa, config);
                      }
                      catch (const std::exception &exc)
                      {
                        out.has_error = true;
                        out.error_message = exc.what();
                      }
                      catch (...)
                      {
                        out.has_error = true;
                        out.error_message = "unknown error in process_plane_core";
                      }
                    });

  return results;
}
