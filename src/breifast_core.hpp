// Python/pybind11-independent core of breifast_helpers: the actual per-band-slice detection
// algorithm, usable both from the pybind11 extension (breifast_helpers.cpp) and from a plain
// C++ standalone driver (bench_process_band_slice.cpp) for profiling with tools like VTune
// without any Python runtime involved.
#pragma once

#define XTENSOR_USE_TBB
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>
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
 * performance/precision -- see bench_process_band_slice.cpp.
 *
 * @tparam Accum Accumulation type (`double` by default).
 * @param img (H, W) input image.
 * @return (H+1, W+1) summed-area table, in `Accum` precision.
 */
template <typename Accum = double>
inline xt::xarray<Accum> compute_integral_image(const xt::xarray<float> &img)
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
 * (bench_process_band_slice.cpp) -- see that file's docstring for the algorithm description.
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
    const xt::xarray<float> &channel_width, long band_start, long band_stop, bool is_single_channel,
    bool subtract_mean, const std::optional<xt::xarray<float>> &mean_image, bool use_rms_noise_estimator,
    int boxsize, const xt::xarray<float> &selfnoise_offset, const xt::xarray<float> &dde_noise,
    float ddenoise_scaling_factor, float full_bandwidth, const std::optional<xt::xarray<float>> &flux_cutoff,
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

  // SNSF inflates the selfnoise term below for bands narrower than the full bandwidth -- less spectral averaging means
  // proportionally more noise
  const float bandwidth = xt::sum(xt::xarray<float>(xt::view(channel_width, xt::range(band_start, band_stop))))();
  const float snsf = full_bandwidth / bandwidth;
  // materialize abs(plane_img) before handing it to xt::median(): passed as a lazy expression, median's
  // selection algorithm walks it via xtensor's generic (slow, unvectorized) stepper, re-evaluating abs_fun on
  // every element access during its several non-sequential passes -- measured as ~99% of total CPU time in a
  // VTune profile of this function. Assigning into a concrete xt::xarray first evaluates it once via the fast,
  // TBB-parallelized linear_assigner path instead.
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
                                                 snr_cutoff, flux_cutoff, reject_noise_threshold, band_start,
                                                 band_stop, band_length, histogram_detail, max_sigma_bin);
  image_mask = std::move(cutoffs.image_mask);
  std::vector<long> hist1 = std::move(cutoffs.hist1);
  std::vector<long> hist2 = std::move(cutoffs.hist2);

  // Early exit if all pixels are masked
  if (!xt::any(image_mask))
  {
    ProcessBandSliceResult result;
    result.processed = false;
    result.mednoise = mednoise;
    result.noise_globmad = noise_globmad;
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

  // Detection threshold application
  abs_snr = xt::where(image_mask, abs_snr, 0.f);
  xt::xarray<bool> final_mask = abs_snr >= threshold;

  // Write back SNR values into the cube
  if (is_single_channel)
  {
    xt::view(stokes_cube, band_start, xt::all(), xt::all()) = snr;
  }

  ProcessBandSliceResult result;
  result.processed = true;
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
