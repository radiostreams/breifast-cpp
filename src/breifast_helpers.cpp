#define XTENSOR_USE_TBB
#define FORCE_IMPORT_ARRAY
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include <pybind11/pybind11.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/reducers/xreducer.hpp>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <xtensor-python/pyarray.hpp>

float compute_sumww(
    xt::pyarray<float> &weights,
    size_t stokes_index,
    const pybind11::slice &band_slice,
    size_t interval)
{
  Py_ssize_t start, stop, step, length;
  band_slice.compute(weights.shape(1), &start, &stop, &step, &length);

  const auto ww = xt::view(weights,
                           stokes_index,
                           xt::range(start, stop),
                           interval);

  return xt::sum(ww)();
}

// Port of breifast.py's max_to_std_ratio(): finds the value x such that
// F(x) = 1 - (1 - Phi(x))^n crosses 0.5, i.e. the expected ratio of the
// maximum of n standard-normal draws to their standard deviation.
// Results are cached by n, matching the Python-side memoization.
double max_to_std_ratio(double n)
{
  static std::unordered_map<double, double> cache;
  static std::mutex cache_mutex;

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(n);
    if (it != cache.end())
      return it->second;
  }

  constexpr int num_points = 1000;
  constexpr double x0 = -10.0, x1 = 10.0;

  double prev_x = x0;
  double prev_F = 0.0;
  double result = x1;
  bool found = false;

  for (int i = 0; i < num_points; ++i)
  {
    const double x = x0 + (x1 - x0) * i / (num_points - 1);
    const double f = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    const double F = 1.0 - std::pow(1.0 - f, n);

    if (!found && F >= 0.5)
    {
      if (i == 0)
      {
        result = x;
      }
      else
      {
        const double t = (0.5 - prev_F) / (F - prev_F);
        result = prev_x + t * (x - prev_x);
      }
      found = true;
    }
    prev_x = x;
    prev_F = F;
  }

  result = std::abs(result);

  std::lock_guard<std::mutex> lock(cache_mutex);
  cache[n] = result;
  return result;
}

// Builds a summed-area table (with an extra leading row/column of zeros) for
// fast box sums, used to implement the sliding uniform_filter box below.
xt::xarray<double> compute_integral_image(const xt::xarray<float> &img)
{
  const size_t h = img.shape(0), w = img.shape(1);
  xt::xarray<double> integral = xt::zeros<double>({h + 1, w + 1});
  for (size_t i = 0; i < h; ++i)
  {
    double row_sum = 0.0;
    for (size_t j = 0; j < w; ++j)
    {
      row_sum += static_cast<double>(img(i, j));
      integral(i + 1, j + 1) = integral(i, j + 1) + row_sum;
    }
  }
  return integral;
}

inline double box_sum(const xt::xarray<double> &integral,
                       long r0, long r1, long c0, long c1,
                       long h, long w)
{
  r0 = std::clamp(r0, 0L, h);
  r1 = std::clamp(r1, 0L, h);
  c0 = std::clamp(c0, 0L, w);
  c1 = std::clamp(c1, 0L, w);
  if (r0 >= r1 || c0 >= c1)
    return 0.0;
  return integral(r1, c1) - integral(r0, c1) - integral(r1, c0) + integral(r0, c0);
}

// Port of breifast.py's make_noise_map(..., method='std') -- the only branch
// used by process_plane. Reproduces scipy.ndimage.uniform_filter(mode='constant')
// via a summed-area table: a fixed-size box average with zero-padding outside
// the image, normalized by the true (possibly partial) footprint size to
// correct for edge effects, exactly like the Python "scale" division.
std::pair<xt::xarray<float>, float> make_noise_map(const xt::xarray<float> &plane_img, long boxsize)
{
  const long h = static_cast<long>(plane_img.shape(0));
  const long w = static_cast<long>(plane_img.shape(1));
  const long lo = boxsize / 2;
  const double box_area = static_cast<double>(boxsize) * static_cast<double>(boxsize);

  xt::xarray<float> img_sq = plane_img * plane_img;
  const auto integral_sq = compute_integral_image(img_sq);

  xt::xarray<float> noise = xt::zeros<float>({static_cast<size_t>(h), static_cast<size_t>(w)});

  for (long i = 0; i < h; ++i)
  {
    const long r0 = i - lo, r1 = r0 + boxsize;
    for (long j = 0; j < w; ++j)
    {
      const long c0 = j - lo, c1 = c0 + boxsize;
      const double sum2 = box_sum(integral_sq, r0, r1, c0, c1, h, w);
      const double valid_count = static_cast<double>(std::max(0L, std::min(r1, h) - std::max(r0, 0L))) *
                                  static_cast<double>(std::max(0L, std::min(c1, w) - std::max(c0, 0L)));
      const double scale = std::sqrt(valid_count / box_area);
      const double raw_noise = std::sqrt(sum2 / box_area);
      noise(i, j) = scale != 0.0 ? static_cast<float>(raw_noise / scale) : 0.0f;
    }
  }

  const float median_noise = xt::median(noise);
  xt::filter(noise, noise < median_noise) = median_noise;

  return {noise, median_noise};
}

// Fixed-width histogram matching np.histogram(data, np.arange(-max_sigma_bin, max_sigma_bin + 1))
// i.e. integer bin edges of width 1, values outside [-max_sigma_bin, max_sigma_bin] dropped.
std::vector<long> histogram_counts(const xt::xarray<float> &values, const xt::xarray<bool> &mask, int max_sigma_bin)
{
  const int nbins = 2 * max_sigma_bin;
  std::vector<long> counts(nbins, 0);
  auto it_v = values.begin();
  auto it_m = mask.begin();
  for (; it_v != values.end(); ++it_v, ++it_m)
  {
    if (!*it_m)
      continue;
    const float v = *it_v;
    if (v < -max_sigma_bin || v > max_sigma_bin)
      continue;
    int idx = static_cast<int>(std::floor(v + max_sigma_bin));
    if (idx >= nbins)
      idx = nbins - 1;
    if (idx < 0)
      idx = 0;
    counts[idx]++;
  }
  return counts;
}

// Port of the per-band-slice body of process_plane() in breifast.py (roughly
// lines 633-741): forms the (possibly weighted-mean) plane image for one band
// slice, estimates the local noise map, converts to SNR units, and applies
// the snr/flux/local-noise rejection masks, producing the final
// threshold mask. Island detection and per-island statistics (scipy.ndimage.label
// / find_objects and everything downstream) are NOT ported yet -- xtensor has
// no connected-component labeling primitive, so that part still needs to run
// in Python (or a future dedicated C++ implementation) using the mask/snr/
// plane_img/noisemap/excess_noise_factor returned here.
pybind11::tuple process_band_slice(
    xt::pyarray<float> &stokes_cube,           // (FREQ, H, W) for this stokes+interval; mutated in place if is_single_channel
    xt::pyarray<float> &weights,                // (FREQ,) for this stokes+interval
    xt::pyarray<bool> &accept_mask,             // (FREQ, H, W) for this stokes
    xt::pyarray<float> &channel_width,          // (FREQ,) full per-channel widths
    const pybind11::slice &band_slice,
    bool is_single_channel,
    bool subtract_mean,
    pybind11::object mean_image_obj,            // (FREQ, H, W) for this stokes, or None
    bool use_rms_noise_estimator,               // true: 'rms', false: 'gmad'
    int boxsize,
    xt::pyarray<float> &selfnoise_offset,       // (H, W) for this stokes
    xt::pyarray<float> &dde_noise,              // (H, W) for this stokes
    float ddenoise_scaling_factor,
    float full_bandwidth,
    pybind11::object flux_cutoff_obj,           // (FREQ, H, W) for this stokes, or None
    pybind11::object snr_cutoff_obj,            // (FREQ, H, W) for this stokes, or None
    float reject_noise_threshold,
    int histogram_detail,
    int max_sigma_bin,
    float threshold)
{
  const auto freq_len = stokes_cube.shape(0);

  Py_ssize_t bstart, bstop, bstep, blen;
  band_slice.compute(freq_len, &bstart, &bstop, &bstep, &blen);
  if (bstep != 1)
    throw std::invalid_argument("process_band_slice: only unit-step band slices are supported");

  // xt::range(start, stop) is constructed fresh at each xt::view() call site
  // (rather than stored in a variable and reused) to match the type xt::view's
  // internal slicing machinery expects -- storing it in a variable first and
  // passing that around changes its deduced type (xrange vs xrange_adaptor)
  // and xt::view() rejects the mismatch.
  const xt::xarray<float> ww = xt::view(weights, xt::range(bstart, bstop));
  const double sumww = xt::sum(ww)();

  // form the plane image: single channel is a plain view, otherwise the
  // weighted mean over the band slice
  xt::xarray<float> plane_img;
  if (is_single_channel)
  {
    plane_img = xt::view(stokes_cube, bstart, xt::all(), xt::all());
  }
  else
  {
    const xt::xarray<float> band = xt::view(stokes_cube, xt::range(bstart, bstop), xt::all(), xt::all());
    const auto weighted = band * xt::view(ww, xt::all(), xt::newaxis(), xt::newaxis());
    plane_img = xt::sum(weighted, {0}) / sumww;
  }

  const xt::xarray<bool> band_accept_mask = xt::view(accept_mask, xt::range(bstart, bstop), xt::all(), xt::all());
  // xtensor has no axis-reducing xt::any(); xt::reduce is its documented
  // mechanism for applying a reducing function over given axes, so reducing
  // with logical_or along axis 0 is the exact equivalent of numpy's any(axis=0)
  xt::xarray<bool> image_mask = xt::reduce(std::logical_or<bool>(), band_accept_mask, {0});

  // matches Python's `if (plane_img == 0).all() or not image_mask.any(): continue`
  // (before any noise stats have been computed, so no write-back needed here)
  const bool empty_plane = xt::all(xt::equal(plane_img, 0.f));
  if (empty_plane || !xt::any(image_mask))
  {
    return pybind11::make_tuple(false, pybind11::none(), pybind11::none(), pybind11::none(),
                                 pybind11::none(), pybind11::none(), pybind11::none(),
                                 pybind11::none(), pybind11::none(), pybind11::none(),
                                 pybind11::none(), pybind11::none(), pybind11::none());
  }

  if (subtract_mean && !mean_image_obj.is_none())
  {
    const auto mean_image = mean_image_obj.cast<xt::pyarray<float>>();
    const xt::xarray<float> band_mean = xt::view(mean_image, xt::range(bstart, bstop), xt::all(), xt::all());
    plane_img = plane_img - xt::sum(band_mean, {0}) / static_cast<float>(band_mean.shape(0));
  }

  const double bandwidth = xt::sum(xt::xarray<float>(xt::view(channel_width, xt::range(bstart, bstop))))();
  const double snsf = full_bandwidth / bandwidth;

  const float noise_globmad = static_cast<float>(xt::median(xt::abs(plane_img))) * 1.4826f;

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

  xt::xarray<float> noisemap = xt::sqrt(
      noisemap0 * noisemap0 + static_cast<float>(snsf) * selfnoise_offset +
      ddenoise_scaling_factor * dde_noise);

  xt::xarray<float> excess_noise_factor = xt::where(
      xt::not_equal(noisemap0, 0.f), noisemap / noisemap0, 0.f);

  const float mednoise = static_cast<float>(xt::median(noisemap));

  xt::xarray<float> snr = xt::where(xt::not_equal(noisemap, 0.f), plane_img / noisemap, 0.f);
  const xt::xarray<bool> valid_noisemap = xt::not_equal(noisemap, 0.f) & xt::isfinite(noisemap);
  snr = xt::where(valid_noisemap, snr, 0.f);
  xt::xarray<float> abs_snr = xt::abs(snr);
  image_mask = image_mask & valid_noisemap;

  std::vector<long> hist0, hist1, hist2;
  if (histogram_detail > 1)
    hist0 = histogram_counts(snr, image_mask, max_sigma_bin);

  if (!snr_cutoff_obj.is_none())
  {
    const auto snr_cutoff = snr_cutoff_obj.cast<xt::pyarray<float>>();
    const xt::xarray<float> band_snr_cutoff_sum = xt::sum(
        xt::xarray<float>(xt::view(snr_cutoff, xt::range(bstart, bstop), xt::all(), xt::all())), {0});
    const xt::xarray<float> band_snr_cutoff = band_snr_cutoff_sum / static_cast<float>(blen);
    image_mask = image_mask & (abs_snr >= band_snr_cutoff);
  }

  if (histogram_detail > 1)
    hist1 = histogram_counts(snr, image_mask, max_sigma_bin);

  if (!flux_cutoff_obj.is_none())
  {
    const auto flux_cutoff = flux_cutoff_obj.cast<xt::pyarray<float>>();
    const xt::xarray<float> band_flux_cutoff_sum = xt::sum(
        xt::xarray<float>(xt::view(flux_cutoff, xt::range(bstart, bstop), xt::all(), xt::all())), {0});
    const xt::xarray<float> band_flux_cutoff = band_flux_cutoff_sum / static_cast<float>(blen);
    image_mask = image_mask & (xt::abs(plane_img) >= band_flux_cutoff);
  }

  if (histogram_detail > 0)
    hist2 = histogram_counts(snr, image_mask, max_sigma_bin);

  if (reject_noise_threshold != 0.f)
  {
    image_mask = image_mask & (noisemap <= mednoise * reject_noise_threshold);
  }

  if (!xt::any(image_mask))
  {
    return pybind11::make_tuple(false, pybind11::none(), pybind11::none(), pybind11::none(),
                                 pybind11::none(), pybind11::none(), mednoise, noise_globmad,
                                 pybind11::none(), pybind11::none(), pybind11::none(),
                                 pybind11::none(), pybind11::none());
  }

  // minsig/maxsig over the accepted pixels
  float minsig = std::numeric_limits<float>::infinity();
  float maxsig = -std::numeric_limits<float>::infinity();
  {
    auto it_v = snr.begin();
    auto it_m = image_mask.begin();
    for (; it_v != snr.end(); ++it_v, ++it_m)
    {
      if (!*it_m)
        continue;
      minsig = std::min(minsig, *it_v);
      maxsig = std::max(maxsig, *it_v);
    }
  }

  abs_snr = xt::where(image_mask, abs_snr, 0.f);
  xt::xarray<bool> final_mask = abs_snr >= threshold;

  // side effect matching Python's `finally: if is_single_chan: plane_img[:] = snr`
  if (is_single_channel)
  {
    xt::view(stokes_cube, bstart, xt::all(), xt::all()) = snr;
  }

  // xtensor-python only registers pybind11 casters for xt::pyarray/xt::pytensor,
  // not for plain xt::xarray, so results are wrapped before crossing into Python;
  // make_tuple applies the registered caster to each argument itself, so no
  // explicit pybind11::cast() call is needed (and its free-function overload
  // set doesn't resolve cleanly for xt::pyarray anyway)
  return pybind11::make_tuple(
      true,
      xt::pyarray<float>(snr),
      xt::pyarray<bool>(final_mask),
      xt::pyarray<float>(plane_img),
      xt::pyarray<float>(noisemap),
      xt::pyarray<float>(excess_noise_factor),
      mednoise,
      noise_globmad,
      minsig,
      maxsig,
      histogram_detail > 1 ? pybind11::cast(hist0) : pybind11::none(),
      histogram_detail > 1 ? pybind11::cast(hist1) : pybind11::none(),
      histogram_detail > 0 ? pybind11::cast(hist2) : pybind11::none());
}

PYBIND11_MODULE(breifast_helpers, m)
{
  xt::import_numpy();
  m.def("compute_sumww", &compute_sumww,
        pybind11::arg("weights"),
        pybind11::arg("stokes_index"),
        pybind11::arg("band_slice"),
        pybind11::arg("interval"));

  m.def("max_to_std_ratio", &max_to_std_ratio,
        pybind11::arg("n"));

  m.def("process_band_slice", &process_band_slice,
        pybind11::arg("stokes_cube"),
        pybind11::arg("weights"),
        pybind11::arg("accept_mask"),
        pybind11::arg("channel_width"),
        pybind11::arg("band_slice"),
        pybind11::arg("is_single_channel"),
        pybind11::arg("subtract_mean"),
        pybind11::arg("mean_image") = pybind11::none(),
        pybind11::arg("use_rms_noise_estimator") = true,
        pybind11::arg("boxsize"),
        pybind11::arg("selfnoise_offset"),
        pybind11::arg("dde_noise"),
        pybind11::arg("ddenoise_scaling_factor") = 1.0f,
        pybind11::arg("full_bandwidth"),
        pybind11::arg("flux_cutoff") = pybind11::none(),
        pybind11::arg("snr_cutoff") = pybind11::none(),
        pybind11::arg("reject_noise_threshold") = 0.0f,
        pybind11::arg("histogram_detail") = 0,
        pybind11::arg("max_sigma_bin") = 200,
        pybind11::arg("threshold"));
}

/*class Slice {
public:
  Slice(pybind11::slice slice, long length) : length_(length) {
    auto get = [&](pybind11::handle obj, long default_val) -> long {
      if (obj.is_none())
        return default_val;
      long v = obj.cast<long>();
      return v;
    };

    step_ = get(slice.attr("step"), 1);
    if (step_ == 0)
      throw std::runtime_error("slice step cannot be zero");

    long default_start = (step_ > 0) ? 0 : length_ - 1;
    long default_stop = (step_ > 0) ? length_ : -1;

    start_ = normalize(get(slice.attr("start"), default_start));
    stop_ = normalize(get(slice.attr("stop"), default_stop));
  }

  long start() const { return start_; }
  long stop() const { return stop_; }
  long step() const { return step_; }

private:
  long normalize(long index) const {
    if (index < 0)
      index += length_;
    // For stop with negative step, allow -1; for positive step, clamp to [0,
    // length]
    if (step_ > 0) {
      if (index < 0)
        index = 0;
      if (index > length_)
        index = length_;
    } else {
      // For step < 0, valid stop can be -1 (before first) or length-1..0
      if (index < -1)
        index = -1;
      if (index > length_ - 1)
        index = length_ - 1;
    }
    return index;
  }

  long start_, stop_, step_, length_;
};

class Py3DArrayWrapper {
public:
  Py3DArrayWrapper(pybind11::array_t<double> array) {
    auto buf = array.request();
    if (buf.ndim != 3)
      throw std::runtime_error("Expected a 3D array");
    data_ = static_cast<double *>(buf.ptr);
    shape_ = {buf.shape[0], buf.shape[1], buf.shape[2]};
    strides_ = {buf.strides[0] / sizeof(double),
                buf.strides[1] / sizeof(double),
                buf.strides[2] / sizeof(double)};
  }

  double *data() const { return data_; }
  const std::array<size_t, 3> &shape() const { return shape_; }
  const std::array<long, 3> &strides() const { return strides_; }

private:
  double *data_;
  std::array<size_t, 3> shape_;
  std::array<long, 3> strides_;
};

double sum_weights(pybind11::array_t<double> weights, int stokes_index,
                   const pybind11::slice &band_slice, int interval) {
  Py3DArrayWrapper w(weights);
  Slice s(band_slice, w.shape()[1]);

  // Base pointer for fixed stokes and interval
  double *base =
      w.data() + stokes_index * w.strides()[0] + interval * w.strides()[2];
  long stride_band = w.strides()[1];

  double sum = 0.0;
  long b;
  if (s.step() > 0) {
#pragma omp parallel for private(b) shared(stride_band) reduction(+ : sum)
    for (b = s.start(); b < s.stop(); b += s.step()) {
      sum += base[b * stride_band];
    }
  } else {
#pragma omp parallel for private(b) shared(stride_band) reduction(+ : sum)
    for (b = s.start(); b > s.stop(); b += s.step()) {
      sum += base[b * stride_band];
    }
  }
  return sum;
}*/

