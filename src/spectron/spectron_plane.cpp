#include "spectron_plane.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>


namespace breifast::spectron
{
namespace
{
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

inline double f32(double v)
{
  return static_cast<double>(static_cast<float>(v));
}

/// Python's round() of a float: half to even.
inline long py_round(double v)
{
  return static_cast<long>(std::nearbyint(v));
}

/// numpy's pairwise summation (as add.reduce does over a contiguous run), in T.
template <typename T> T pairwise_sum(const T *a, size_t n)
{
  if (n < 8)
  {
    T res = 0;
    for (size_t i = 0; i < n; ++i)
      res += a[i];
    return res;
  }
  if (n <= 128)
  {
    T r[8];
    for (int j = 0; j < 8; ++j)
      r[j] = a[j];
    size_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j)
        r[j] += a[i + j];
    T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i)
      res += a[i];
    return res;
  }
  size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

/// np.median of `v` (destroyed), in T: NaN if any NaN, the mean of the two middle values for an even count.
template <typename T> double np_median(std::vector<T> &v)
{
  const size_t n = v.size();
  if (!n)
    return NaN;
  for (const T x : v)
    if (std::isnan(x))
      return NaN;
  const size_t k = n / 2;
  std::nth_element(v.begin(), v.begin() + k, v.end());
  const T hi = v[k];
  if (n % 2)
    return hi;
  const T lo = *std::max_element(v.begin(), v.begin() + k);
  return static_cast<T>((lo + hi) / static_cast<T>(2));
}

/// np.std of float32 values (ddof 0), computed as numpy does: float32 pairwise sums.
double np_std_f32(const std::vector<float> &d)
{
  if (d.empty())
    return NaN;
  const float n = static_cast<float>(d.size());
  const float mean = pairwise_sum(d.data(), d.size()) / n;
  std::vector<float> sq(d.size());
  for (size_t i = 0; i < d.size(); ++i)
  {
    const float x = d[i] - mean;
    sq[i] = x * x;
  }
  return std::sqrt(pairwise_sum(sq.data(), sq.size()) / n);
}

/// Mean over the slice's usable bands of a (band, y, x) array, axis 0, as numpy computes it in T.
template <typename T>
void band_mean(const BandArray<T> &arr, const std::vector<long> &bands, size_t npix, std::vector<double> &out)
{
  out.resize(npix);
  thread_local std::vector<T> acc;
  acc.resize(npix);
  const T *p0 = arr.plane(bands[0]);
  std::copy(p0, p0 + npix, acc.begin());
  for (size_t k = 1; k < bands.size(); ++k)
  {
    const T *p = arr.plane(bands[k]);
    for (size_t i = 0; i < npix; ++i)
      acc[i] += p[i];
  }
  const T cnt = static_cast<T>(bands.size());
  for (size_t i = 0; i < npix; ++i)
    out[i] = static_cast<T>(acc[i] / cnt);
}

void band_mean_any(const BandArrayF &arr, const std::vector<long> &bands, size_t npix, std::vector<double> &out)
{
  if (arr.is_f64())
    band_mean(arr.f64, bands, npix, out);
  else
    band_mean(arr.f32, bands, npix, out);
}

/// 4-connected labelling as scipy.ndimage.label: labels in raster order of each island's first pixel.
struct Labels
{
  std::vector<int32_t> lab;
  int32_t n = 0;
};

void label4(const std::vector<uint8_t> &mask, long ny, long nx, Labels &out)
{
  const size_t npix = static_cast<size_t>(ny) * nx;
  out.lab.assign(npix, 0);
  thread_local std::vector<int32_t> parent;
  parent.assign(1, 0);
  const auto find = [&](int32_t a)
  {
    while (parent[a] != a)
    {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  auto &lab = out.lab;
  for (long y = 0; y < ny; ++y)
  {
    for (long x = 0; x < nx; ++x)
    {
      const size_t i = static_cast<size_t>(y) * nx + x;
      if (!mask[i])
        continue;
      const int32_t up = y > 0 ? lab[i - nx] : 0;
      const int32_t left = x > 0 ? lab[i - 1] : 0;
      if (!up && !left)
      {
        const int32_t l = static_cast<int32_t>(parent.size());
        parent.push_back(l);
        lab[i] = l;
      }
      else if (up && left)
      {
        const int32_t ru = find(up), rl = find(left);
        const int32_t r = std::min(ru, rl);
        parent[ru] = parent[rl] = r;
        lab[i] = r;
      }
      else
        lab[i] = up ? up : left;
    }
  }
  // compact ids in raster order of first appearance
  thread_local std::vector<int32_t> compact;
  compact.assign(parent.size(), 0);
  int32_t n = 0;
  for (size_t i = 0; i < npix; ++i)
  {
    if (!lab[i])
      continue;
    const int32_t r = find(lab[i]);
    if (!compact[r])
      compact[r] = ++n;
    lab[i] = compact[r];
  }
  out.n = n;
}

/// Gaussian kernel as cubes.create_multidimensional_gaussian_kernel(sigma, [size]*2, rotate, centred=True), odd size.
/// `sigma_f32`: the sigmas came as a float32 array, so 2*sig**2 is computed in float32.
std::vector<double> gaussian_kernel(long size, double sig0, double sig1, double rotate, bool sigma_f32)
{
  const long h = size / 2;
  std::vector<double> k(static_cast<size_t>(size) * size);
  const double c = std::cos(rotate), s = std::sin(rotate);
  double d0, d1;
  if (sigma_f32)
  {
    const float a = static_cast<float>(sig0), b = static_cast<float>(sig1);
    d0 = 2.F * (a * a);
    d1 = 2.F * (b * b);
  }
  else
  {
    d0 = 2 * (sig0 * sig0);
    d1 = 2 * (sig1 * sig1);
  }
  for (long i = 0; i < size; ++i)
    for (long j = 0; j < size; ++j)
    {
      double x0 = static_cast<double>(i - h), x1 = static_cast<double>(j - h);
      if (rotate != 0.)
      {
        const double r0 = c * x0 + (-s) * x1;
        const double r1 = s * x0 + c * x1;
        x0 = r0;
        x1 = r1;
      }
      const double e0 = sig0 != 0. ? x0 * x0 / d0 : 0.;
      const double e1 = sig1 != 0. ? x1 * x1 / d1 : 0.;
      k[static_cast<size_t>(i) * size + j] = std::exp(-(e0 + e1));
    }
  const double sum = pairwise_sum(k.data(), k.size());
  for (auto &v : k)
    v /= sum;
  return k;
}

/// The (2h+1)^2 window of `plane` centred on (cy, cx), zeros outside: a slice of procres' zero-padded `padimg`.
void stamp(const std::vector<float> &plane, long ny, long nx, long cy, long cx, long h, std::vector<float> &out)
{
  const long w = 2 * h + 1;
  out.assign(static_cast<size_t>(w) * w, 0.F);
  for (long i = 0; i < w; ++i)
  {
    const long y = cy - h + i;
    if (y < 0 || y >= ny)
      continue;
    for (long j = 0; j < w; ++j)
    {
      const long x = cx - h + j;
      if (x >= 0 && x < nx)
        out[static_cast<size_t>(i) * w + j] = plane[static_cast<size_t>(y) * nx + x];
    }
  }
}

double sign(double v)
{
  return v > 0 ? 1. : (v < 0 ? -1. : (v == 0 ? 0. : NaN));
}

/// Pearson correlation as np.corrcoef(a, b)[0, 1], clipped to [-1, 1].
double corrcoef(const std::vector<double> &a, const std::vector<double> &b)
{
  const size_t n = a.size();
  double ma = 0, mb = 0;
  for (size_t i = 0; i < n; ++i)
  {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  double sab = 0, saa = 0, sbb = 0;
  for (size_t i = 0; i < n; ++i)
  {
    const double da = a[i] - ma, db = b[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  const double r = sab / std::sqrt(saa * sbb);
  return std::isnan(r) ? r : std::clamp(r, -1., 1.);
}

/// L1 dilation of `m` (w x w) by `k` steps, as scipy binary_dilation(m, iterations=k) with the default cross.
std::vector<uint8_t> dilate_l1(const std::vector<uint8_t> &m, long w, long k)
{
  const long big = 1L << 40;
  std::vector<long> d(m.size());
  for (size_t i = 0; i < m.size(); ++i)
    d[i] = m[i] ? 0 : big;
  for (long y = 0; y < w; ++y)
    for (long x = 0; x < w; ++x)
    {
      long &v = d[y * w + x];
      if (y > 0)
        v = std::min(v, d[(y - 1) * w + x] + 1);
      if (x > 0)
        v = std::min(v, d[y * w + x - 1] + 1);
    }
  for (long y = w - 1; y >= 0; --y)
    for (long x = w - 1; x >= 0; --x)
    {
      long &v = d[y * w + x];
      if (y < w - 1)
        v = std::min(v, d[(y + 1) * w + x] + 1);
      if (x < w - 1)
        v = std::min(v, d[y * w + x + 1] + 1);
    }
  std::vector<uint8_t> out(m.size());
  for (size_t i = 0; i < m.size(); ++i)
    out[i] = d[i] <= k;
  return out;
}

/// scipy.ndimage.uniform_filter1d(mode='constant', cval=0) along a line, as scipy computes it: a double running sum,
/// updated by (in[i + size - 1] - in[i - 1]) / size, the window [i - size/2, i + size - 1 - size/2].
/// `at(k)` reads input k of the line (0 outside), `put(k, v)` stores output k.
template <typename In, typename Out>
void uniform_line(long length, long size, const In &at, const Out &put)
{
  const long size1 = size / 2;
  const auto ext = [&](long j) { return (j - size1 >= 0 && j - size1 < length) ? at(j - size1) : 0.0; };
  double tmp = 0.0;
  for (long k = 0; k < size; ++k)
    tmp += ext(k);
  tmp /= static_cast<double>(size);
  put(0, tmp);
  for (long l = 1; l < length; ++l)
  {
    tmp += (ext(l + size - 1) - ext(l - 1)) / static_cast<double>(size);
    put(l, tmp);
  }
}

/// breifast.make_noise_map(plane, boxsize, method='std') into `noise` (float32): sqrt(uniform_filter(plane^2)) over
/// sqrt(uniform_filter(ones)), both filters as scipy runs them (axis 0, then axis 1, each pass stored in the array's
/// type), then the median floor.
void make_noise_map_std(const std::vector<float> &plane, long ny, long nx, long box, std::vector<float> &noise,
                        std::vector<float> &scratch)
{
  const size_t npix = static_cast<size_t>(ny) * nx;
  thread_local std::vector<float> sq, pass0;
  sq.resize(npix);
  for (size_t i = 0; i < npix; ++i)
    sq[i] = plane[i] * plane[i];
  pass0.resize(npix);
  for (long x = 0; x < nx; ++x)
    uniform_line(ny, box, [&](long k) { return static_cast<double>(sq[static_cast<size_t>(k) * nx + x]); },
                 [&](long k, double v) { pass0[static_cast<size_t>(k) * nx + x] = static_cast<float>(v); });
  noise.resize(npix);
  for (long y = 0; y < ny; ++y)
  {
    const float *row = pass0.data() + static_cast<size_t>(y) * nx;
    float *out = noise.data() + static_cast<size_t>(y) * nx;
    uniform_line(nx, box, [&](long k) { return static_cast<double>(row[k]); },
                 [&](long k, double v) { out[k] = std::sqrt(static_cast<float>(v)); });
  }
  // the edge scale, from a float64 array of ones
  std::vector<double> s0(ny);
  uniform_line(ny, box, [&](long) { return 1.0; }, [&](long k, double v) { s0[k] = v; });
  thread_local std::vector<double> scale;
  scale.resize(npix);
  for (long y = 0; y < ny; ++y)
    uniform_line(nx, box, [&](long) { return s0[y]; },
                 [&](long k, double v) { scale[static_cast<size_t>(y) * nx + k] = std::sqrt(v); });
  for (size_t i = 0; i < npix; ++i)
    noise[i] = scale[i] == 0. ? 0.F : static_cast<float>(static_cast<double>(noise[i]) / scale[i]);
  scratch.assign(noise.begin(), noise.end());
  const float med = static_cast<float>(np_median(scratch));
  if (!std::isnan(med))
    for (auto &v : noise)
      if (v < med)
        v = med;
}

/// np.histogram(values, bins=arange(-M, M+1)) accumulated into `hist`.
inline void hist_add(std::vector<long> &hist, double v, long M)
{
  if (!(v >= -M && v <= M))
    return;
  long b = static_cast<long>(std::floor(v)) + M;
  if (b >= 2 * M)
    b = 2 * M - 1;
  ++hist[b];
}

} // namespace

double max_to_std_ratio(long n)
{
  static std::map<long, double> cache;
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  const auto it = cache.find(n);
  if (it != cache.end())
    return it->second;
  // x = linspace(-10, 10, 1000); F = 1 - (1 - Phi(x))**n; abs(interp(0.5, F, x))
  constexpr int N = 1000;
  std::vector<double> x(N), F(N);
  for (int i = 0; i < N; ++i)
  {
    x[i] = -10. + 20. * i / (N - 1);
    const double f = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
    F[i] = 1.0 - std::pow(1.0 - f, static_cast<double>(n));
  }
  double r;
  if (0.5 <= F[0])
    r = x[0];
  else if (0.5 >= F[N - 1])
    r = x[N - 1];
  else
  {
    int j = static_cast<int>(std::upper_bound(F.begin(), F.end(), 0.5) - F.begin()) - 1;
    const double slope = (x[j + 1] - x[j]) / (F[j + 1] - F[j]);
    r = slope * (0.5 - F[j]) + x[j];
  }
  r = std::abs(r);
  cache.emplace(n, r);
  return r;
}

void process_plane(const Config &cfg, PlaneInputs &in, PlaneResult &out)
{
  const long ny = cfg.ny, nx = cfg.nx, nfreq = cfg.nfreq;
  const size_t npix = static_cast<size_t>(ny) * nx;
  const long num_band_slices = static_cast<long>(cfg.band_slices.size());
  float *const cube = in.cube;
  const auto chan = [&](long b) { return cube + b * in.cube_band_stride; };

  // Stokes P, formed in place in Q's slot
  if (in.cube_u)
  {
    for (long b = 0; b < nfreq; ++b)
    {
      float *q = chan(b);
      const float *u = in.cube_u + b * in.cube_u_band_stride;
      for (size_t i = 0; i < npix; ++i)
        q[i] = std::sqrt(q[i] * q[i] + u[i] * u[i]);
    }
  }

  out.mednoises.assign(num_band_slices, 0.);
  out.usable.assign(nfreq, 0);
  for (long b = 0; b < nfreq; ++b)
  {
    const float *p = chan(b);
    bool finite = false;
    for (size_t i = 0; i < npix && !finite; ++i)
      finite = std::isfinite(p[i]);
    out.usable[b] = finite && in.nonzero_gate[b];
    out.live_without_finite += in.nonzero_gate[b] && !finite;
  }
  if (std::none_of(out.usable.begin(), out.usable.end(), [](uint8_t v) { return v; }))
  {
    out.returned_early = true;
    return;
  }

  const bool so_f64 = in.selfnoise_offset_f64 != nullptr;
  const bool dd_f64 = in.dde_noise_f64 != nullptr;
  const auto so_at = [&](size_t i) { return so_f64 ? in.selfnoise_offset_f64[i] : in.selfnoise_offset_f32[i]; };
  const auto dd_at = [&](size_t i) { return dd_f64 ? in.dde_noise_f64[i] : in.dde_noise_f32[i]; };
  const long M = cfg.max_sigma_bin;
  const bool rms_mode = cfg.noise_estimator == Config::NoiseEstimator::rms;

  // scratch
  // full-image scratch, kept per thread from plane to plane: allocating it afresh for every plane costs more than
  // the plane itself at high thread counts (procres caps malloc arenas and mmaps anything over 1 MB, so every buffer
  // is fresh zeroed pages and an munmap with its TLB shootdowns)
  struct Scratch
  {
    std::vector<float> plane, rms, fscratch;
    std::vector<double> noisemap, snr, nm_here, cutoff, dscratch, iv, abs_snr;
    std::vector<uint8_t> image_mask, valid;
    Labels labels;
  };
  thread_local Scratch tls;
  Scratch &S = tls; // one TLS lookup, not one per access
  auto &plane = S.plane, &rms = S.rms, &fscratch = S.fscratch;
  auto &noisemap = S.noisemap, &snr = S.snr, &nm_here = S.nm_here, &cutoff = S.cutoff, &dscratch = S.dscratch,
       &iv = S.iv, &abs_snr = S.abs_snr;
  auto &image_mask = S.image_mask, &valid = S.valid;
  auto &labels = S.labels;
  plane.resize(npix);
  noisemap.resize(npix);
  snr.resize(npix);
  image_mask.resize(npix);
  std::vector<float> sub, bstamp, pstamp;

  bool have_inv_var = false;
  std::vector<double> inv_var, model_excess;
  bool minsig_set = false;
  double minsig = 0., maxsig = 0.;
  double mednoise = 0.;

  for (long iband = 0; iband < num_band_slices; ++iband)
  {
    const auto &sl = cfg.band_slices[num_band_slices - 1 - iband];
    const bool single = (sl.stop - sl.start) == 1;
    std::vector<long> bands;
    for (long b = sl.start; b < sl.stop; ++b)
      if (out.usable[b])
        bands.push_back(b);
    if (static_cast<long>(bands.size()) < (single ? 1 : 2))
      continue;
    std::vector<float> ww(bands.size());
    float sumww = 0.F;
    for (size_t k = 0; k < bands.size(); ++k)
    {
      ww[k] = in.weights[bands[k]];
      sumww += ww[k];
    }
    float *single_chan = single ? chan(sl.start) : nullptr;
    if (single)
      std::copy(single_chan, single_chan + npix, plane.begin());
    else
    {
      for (size_t k = 0; k < bands.size(); ++k)
      {
        const float *p = chan(bands[k]);
        if (!k)
          for (size_t i = 0; i < npix; ++i)
            plane[i] = p[i] * ww[k];
        else
          for (size_t i = 0; i < npix; ++i)
            plane[i] += p[i] * ww[k];
      }
      for (size_t i = 0; i < npix; ++i)
        plane[i] /= sumww;
    }
    // image_mask = accept_mask[stokes, slice_bands].any(axis=0)
    std::fill(image_mask.begin(), image_mask.end(), 0);
    for (const long b : bands)
    {
      const uint8_t *m = in.accept_mask.plane(b);
      for (size_t i = 0; i < npix; ++i)
        image_mask[i] |= m[i];
    }
    if (std::all_of(plane.begin(), plane.end(), [](float v) { return v == 0.F; }) ||
        std::none_of(image_mask.begin(), image_mask.end(), [](uint8_t v) { return v; }))
      continue;

    if (cfg.subtract_mean)
    {
      band_mean(in.mean_image, bands, npix, dscratch);
      for (size_t i = 0; i < npix; ++i)
        plane[i] -= static_cast<float>(dscratch[i]);
      if (single) // plane_img is a view of the cube there
        std::copy(plane.begin(), plane.end(), single_chan);
    }

    // per-band noise model of the slice
    bool nm_here_is_band = false;
    if (in.nm_noise_band)
    {
      nm_here_is_band = true;
      nm_here.assign(npix, 0.);
      double sw = 0.;
      for (const float w : ww)
        sw += static_cast<double>(w);
      if (sw > 0)
      {
        for (size_t k = 0; k < bands.size(); ++k)
        {
          const double w2 = static_cast<double>(ww[k]) * static_cast<double>(ww[k]);
          const double *nb = in.nm_noise_band.plane(bands[k]);
          for (size_t i = 0; i < npix; ++i)
            nm_here[i] += w2 * nb[i];
        }
        for (size_t i = 0; i < npix; ++i)
          nm_here[i] /= sw * sw;
      }
    }
    const auto nm_at = [&](size_t i)
    {
      if (nm_here_is_band)
        return nm_here[i];
      return in.nm_noise_f64 ? in.nm_noise_f64[i] : static_cast<double>(in.nm_noise_f32[i]);
    };

    // bandwidth and selfnoise scale
    float bandwidth = 0.F;
    for (const long b : bands)
      bandwidth += in.chanwidth[b];
    const bool snsf_f32 = cfg.selfnoise_bandwidth_scaling;
    const double snsf = snsf_f32 ? static_cast<double>(cfg.full_bandwidth / bandwidth) : 1.0;

    // global MAD, float32
    fscratch.resize(npix);
    for (size_t i = 0; i < npix; ++i)
      fscratch[i] = std::abs(plane[i]);
    const float noise_globmad = static_cast<float>(np_median(fscratch)) * 1.4826F;

    // base noise estimate: float32, per pixel (rms) or scalar (MAD)
    if (rms_mode)
    {
      make_noise_map_std(plane, ny, nx, cfg.boxsize, rms, fscratch);
    }
    const auto n0_at = [&](size_t i) -> double { return rms_mode ? rms[i] : noise_globmad; };

    // noisemap = sqrt(noisemap0**2 * (1 + nfe) + snsf * so + dde * dd + nm_here) [* noise_scale_sqrt]
    // in numpy's types: float32 unless a float64 operand enters
    const float one_nfe = static_cast<float>(1.0 + cfg.noise_floor_excess);
    const bool term1_f64 = so_f64;
    const bool term2_f64 = dd_f64;
    const bool nm_is_f64 = nm_here_is_band || in.nm_noise_f64;
    bool noise_f64 = term1_f64 || term2_f64 || nm_is_f64;
    const bool final_f64 = noise_f64 || in.noise_scale_sqrt != nullptr;
    for (size_t i = 0; i < npix; ++i)
    {
      const float n0 = static_cast<float>(n0_at(i));
      double acc = static_cast<float>(static_cast<float>(n0 * n0) * one_nfe);
      double t1 = term1_f64 ? snsf * so_at(i)
                            : (snsf_f32 ? f32(static_cast<float>(snsf) * static_cast<float>(so_at(i)))
                                        : f32(1.0F * static_cast<float>(so_at(i))));
      acc = (term1_f64) ? acc + t1 : f32(acc + t1);
      const double t2 = term2_f64 ? cfg.ddenoise_scaling_factor * dd_at(i)
                                  : f32(static_cast<float>(cfg.ddenoise_scaling_factor) * static_cast<float>(dd_at(i)));
      acc = (term1_f64 || term2_f64) ? acc + t2 : f32(acc + t2);
      const double t3 = nm_at(i);
      acc = noise_f64 ? acc + t3 : f32(acc + t3);
      double nm = noise_f64 ? std::sqrt(acc) : static_cast<double>(std::sqrt(static_cast<float>(acc)));
      if (in.noise_scale_sqrt)
        nm = nm * in.noise_scale_sqrt[i];
      noisemap[i] = nm;
    }
    // excess noise factor at a pixel: noisemap / noisemap0 (the noisemap's type)
    const auto enf_at = [&](size_t i)
    {
      const double v = noisemap[i] / n0_at(i);
      return final_f64 ? v : f32(v);
    };

    // median noise, in the noisemap's type
    if (final_f64)
    {
      dscratch.assign(noisemap.begin(), noisemap.end());
      mednoise = np_median(dscratch);
    }
    else
    {
      fscratch.assign(noisemap.begin(), noisemap.end());
      mednoise = np_median(fscratch);
    }
    // the loop runs over the slices in reverse; mednoises is in forward order
    out.mednoises[num_band_slices - 1 - iband] = mednoise;

    // SNR
    valid.resize(npix);
    for (size_t i = 0; i < npix; ++i)
    {
      valid[i] = noisemap[i] != 0. && std::isfinite(noisemap[i]);
      const double s = static_cast<double>(plane[i]) / noisemap[i];
      snr[i] = valid[i] ? (final_f64 ? s : f32(s)) : 0.;
    }

    // inverse thermal variance and model excess (single-band slices of a multi-band cube)
    if (single && nfreq > 1)
    {
      if (!have_inv_var)
      {
        inv_var.assign(npix, 0.);
        model_excess.assign(npix, 0.);
        have_inv_var = true;
      }
      iv.resize(npix);
      for (size_t i = 0; i < npix; ++i)
      {
        const float n0 = static_cast<float>(n0_at(i));
        iv[i] = (valid[i] && n0 > 0) ? f32(1.0F / (n0 * n0)) : 0.;
        inv_var[i] += iv[i];
        const double t1 = term1_f64 ? snsf * so_at(i) : f32(static_cast<float>(snsf) * static_cast<float>(so_at(i)));
        const double t2 = term2_f64 ? cfg.ddenoise_scaling_factor * dd_at(i)
                                    : f32(static_cast<float>(cfg.ddenoise_scaling_factor) * static_cast<float>(dd_at(i)));
        model_excess[i] += iv[i] * (t1 + t2 + nm_at(i));
      }
      if (in.pb_ivar)
      {
        std::lock_guard<std::mutex> guard(*in.pb_ivar_lock);
        double *acc = in.pb_ivar + static_cast<size_t>(sl.start) * npix;
        for (size_t i = 0; i < npix; ++i)
          acc[i] += iv[i];
      }
    }

    for (size_t i = 0; i < npix; ++i)
      image_mask[i] = image_mask[i] && valid[i];

    // QUIRK: the histograms are reset for every band slice that gets this far, so a plane returns only the last one's
    std::optional<std::vector<long>> hist[3];
    const auto histogram = [&](int which)
    {
      hist[which].emplace(2 * M, 0);
      for (size_t i = 0; i < npix; ++i)
        if (image_mask[i])
          hist_add(*hist[which], snr[i], M);
    };
    if (cfg.histogram_detail > 1)
      histogram(0);
    if (in.snr_cutoff)
    {
      band_mean_any(in.snr_cutoff, bands, npix, cutoff);
      for (size_t i = 0; i < npix; ++i)
        image_mask[i] = image_mask[i] && std::abs(snr[i]) >= cutoff[i];
    }
    if (cfg.histogram_detail > 1)
      histogram(1);
    if (in.flux_cutoff)
    {
      band_mean_any(in.flux_cutoff, bands, npix, cutoff);
      for (size_t i = 0; i < npix; ++i)
        image_mask[i] = image_mask[i] && std::abs(static_cast<double>(plane[i])) >= cutoff[i];
    }
    if (cfg.histogram_detail > 0)
      histogram(2);
    for (int h = 0; h < 3; ++h)
      out.hist[h] = hist[h];
    if (cfg.reject_noise_threshold != 0.)
    {
      const double lim = mednoise * cfg.reject_noise_threshold;
      for (size_t i = 0; i < npix; ++i)
        image_mask[i] = image_mask[i] && noisemap[i] <= lim;
    }
    if (std::none_of(image_mask.begin(), image_mask.end(), [](uint8_t v) { return v; }))
      continue;

    // from here on, a single-band slice is replaced by its SNR image at the end (procres' finally: clause)
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (size_t i = 0; i < npix; ++i)
      if (image_mask[i])
      {
        lo = std::min(lo, snr[i]);
        hi = std::max(hi, snr[i]);
      }
    minsig = minsig_set ? std::min(minsig, lo) : lo;
    maxsig = minsig_set ? std::max(maxsig, hi) : hi;
    minsig_set = true;
    abs_snr.resize(npix);
    for (size_t i = 0; i < npix; ++i)
      abs_snr[i] = image_mask[i] ? std::abs(snr[i]) : 0.;
    const double thr_b = in.slice_threshold[num_band_slices - 1 - iband];

    // QUIRK: the windowed tail calibration labels the whole filtered acceptance mask, not the islands above the
    // nominal threshold that its comment describes
    if (cfg.nominal_peaks && !cfg.skip_detection)
    {
      label4(image_mask, ny, nx, labels);
      if (labels.n)
      {
        std::vector<double> peaks(labels.n, -std::numeric_limits<double>::infinity());
        for (size_t i = 0; i < npix; ++i)
          if (labels.lab[i])
            peaks[labels.lab[i] - 1] = std::max(peaks[labels.lab[i] - 1], abs_snr[i]);
        out.nominal_peaks.emplace_back(sl.label, std::move(peaks));
      }
    }
    if (cfg.skip_detection)
      std::fill(image_mask.begin(), image_mask.end(), 0);
    else
      for (size_t i = 0; i < npix; ++i)
        image_mask[i] = abs_snr[i] >= thr_b;

    if (std::any_of(image_mask.begin(), image_mask.end(), [](uint8_t v) { return v; }))
    {
      label4(image_mask, ny, nx, labels);
      const long nisl = labels.n;
      // island pixel lists (raster order) and bounding boxes
      std::vector<size_t> start(nisl + 1, 0);
      for (size_t i = 0; i < npix; ++i)
        if (labels.lab[i])
          ++start[labels.lab[i]];
      for (long l = 1; l <= nisl; ++l)
        start[l] += start[l - 1];
      std::vector<size_t> pix(start[nisl]);
      {
        std::vector<size_t> pos(start.begin(), start.end() - 1);
        for (size_t i = 0; i < npix; ++i)
          if (labels.lab[i])
            pix[pos[labels.lab[i] - 1]++] = i;
      }
      std::vector<long> y0(nisl, ny), y1(nisl, -1), x0(nisl, nx), x1(nisl, -1);
      for (long l = 0; l < nisl; ++l)
        for (size_t k = start[l]; k < start[l + 1]; ++k)
        {
          const long yy = static_cast<long>(pix[k] / nx), xx = static_cast<long>(pix[k] % nx);
          y0[l] = std::min(y0[l], yy);
          y1[l] = std::max(y1[l], yy + 1);
          x0[l] = std::min(x0[l], xx);
          x1[l] = std::max(x1[l], xx + 1);
        }

      // local flood cap
      std::vector<std::pair<long, long>> cells;
      std::map<std::pair<long, long>, long> cnt;
      bool any_flooded = false;
      std::map<std::pair<long, long>, bool> flooded;
      if (cfg.flood_cell_max && cfg.flood_cell_size && nisl > cfg.flood_cell_max)
      {
        const long cs = cfg.flood_cell_size;
        for (long l = 0; l < nisl; ++l)
        {
          cells.emplace_back(((y0[l] + y1[l]) / 2) / cs, ((x0[l] + x1[l]) / 2) / cs);
          ++cnt[cells.back()];
        }
        for (const auto &[c, n] : cnt)
          if (n > cfg.flood_cell_max)
          {
            flooded[c] = true;
            any_flooded = true;
            ++out.flood_dropped_cells;
          }
      }

      const long r = cfg.boxsize / 2, r1 = cfg.inner_boxsize;
      long bk_halfsize = 0;
      std::vector<double> bmaj_pl, bmin_pl, bpa_pl;
      long pad = r + 1;
      if (cfg.beam_size)
      {
        if (cfg.fixed_beam)
          bk_halfsize = cfg.fixed_halfsize;
        else
        {
          float s = 0.F;
          for (const long b : bands)
            s += in.bmaj[b];
          const float mean = s / static_cast<float>(bands.size());
          bk_halfsize = py_round(static_cast<double>(mean * 3.F)) + 1;
          for (const long b : bands)
          {
            bmaj_pl.push_back(in.bmaj[b]);
            bmin_pl.push_back(in.bmin[b]);
            bpa_pl.push_back(in.bpa[b]);
          }
        }
        pad = std::max(bk_halfsize, r + 1);
      }
      std::vector<double> bk;
      const double ratio = max_to_std_ratio((2 * r + 1) * (2 * r + 1));

      for (long n = 0; n < nisl; ++n)
      {
        long is_flooded = 0;
        if (any_flooded && flooded.count(cells[n]))
        {
          if (cfg.flood_cell_hard && cnt[cells[n]] <= cfg.flood_cell_hard)
            is_flooded = 1;
          else
          {
            ++out.flood_dropped_islands;
            continue;
          }
        }
        // peak |sigma|, the first in raster order
        double best = -1.;
        size_t best_i = pix[start[n]];
        for (size_t k = start[n]; k < start[n + 1]; ++k)
        {
          const double a = std::abs(snr[pix[k]]);
          if (a > best)
          {
            best = a;
            best_i = pix[k];
          }
        }
        maxsig = best; // QUIRK: the island loop reuses the name of the plane's maximum sigma
        const long x = static_cast<long>(best_i % nx), y = static_cast<long>(best_i / nx);
        if (cfg.edge_margin &&
            (x < cfg.edge_margin || y < cfg.edge_margin || x >= nx - cfg.edge_margin || y >= ny - cfg.edge_margin))
        {
          ++out.edge_dropped;
          continue;
        }
        const float flmax = plane[best_i];
        // donut box around the peak, inner box zeroed
        stamp(plane, ny, nx, y, x, r, sub);
        const long w = 2 * r + 1;
        for (long i = std::max(r - r1, 0L); i < std::min(r + r1 + 1, w); ++i)
          for (long j = std::max(r - r1, 0L); j < std::min(r + r1 + 1, w); ++j)
            sub[i * w + j] = 0.F;
        fscratch.clear();
        for (const float v : sub)
          if (v != 0.F)
            fscratch.push_back(v);
        if (cfg.min_sigma_rms != 0.)
        {
          const double rms_d = fscratch.empty() ? 0. : static_cast<double>(static_cast<float>(np_std_f32(fscratch)));
          if (rms_d != 0. && std::abs(static_cast<double>(flmax)) / rms_d < cfg.min_sigma_rms)
          {
            ++out.rms_dropped;
            continue;
          }
        }
        out.flooded.push_back(is_flooded);
        out.flood_tagged += is_flooded;
        out.x.push_back(x);
        out.y.push_back(y);
        out.sigma.push_back(best);
        out.flux.push_back(flmax);

        // ionospheric dipole
        double dsrc = 0., dmir = NaN, dsep = NaN;
        if (!cfg.dip_x.empty() && cfg.beam_size)
        {
          double bm;
          if (cfg.fixed_beam)
            bm = cfg.fixed_bmaj;
          else
            bm = std::accumulate(bmaj_pl.begin(), bmaj_pl.end(), 0.) / bmaj_pl.size();
          const double R = std::max(cfg.dipole_radius * bm, 2.0);
          double dmin = std::numeric_limits<double>::infinity();
          size_t imin = 0;
          for (size_t s = 0; s < cfg.dip_x.size(); ++s)
          {
            const double dx = cfg.dip_x[s] - x, dy = cfg.dip_y[s] - y;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d < dmin)
            {
              dmin = d;
              imin = s;
            }
          }
          if (std::isfinite(dmin) && dmin <= R)
          {
            dsrc = cfg.dip_flux[imin];
            dsep = dmin;
            const long mx = py_round(2 * cfg.dip_x[imin] - x), my = py_round(2 * cfg.dip_y[imin] - y);
            if (my >= 0 && my < ny && mx >= 0 && mx < nx && flmax != 0.F)
            {
              const long hw = std::clamp(py_round(0.6 * dsep), 1L, 4L);
              const double want = -sign(flmax);
              double mv = NaN, mva = -1.;
              bool found = false;
              for (long yy = std::max(my - hw, 0L); yy < std::min(my + hw + 1, ny); ++yy)
                for (long xx = std::max(mx - hw, 0L); xx < std::min(mx + hw + 1, nx); ++xx)
                {
                  const float v = plane[static_cast<size_t>(yy) * nx + xx];
                  if (sign(v) == want && std::abs(v) > mva)
                  {
                    mva = std::abs(v);
                    mv = v;
                    found = true;
                  }
                }
              if (!found)
                mv = plane[static_cast<size_t>(my) * nx + mx];
              dmir = mv / static_cast<double>(flmax);
            }
          }
        }
        out.dip_src.push_back(dsrc);
        out.dip_mirror.push_back(dmir);
        out.dip_sep.push_back(dsep);

        // flux boost against the baseline image: max(flmax / baseline[island], key=abs)
        if (in.baseline_flux)
        {
          float bst = 0.F;
          bool first = true;
          for (size_t k = start[n]; k < start[n + 1]; ++k)
          {
            const float v = flmax / in.baseline_flux[pix[k]];
            if (first || std::abs(v) > std::abs(bst))
              bst = v;
            first = false;
          }
          out.boost.push_back(bst);
        }
        else
          out.boost.push_back(0.);
        out.size.push_back(static_cast<long>(start[n + 1] - start[n]));
        double ln = -std::numeric_limits<double>::infinity();
        for (size_t k = start[n]; k < start[n + 1]; ++k)
          ln = std::max(ln, noisemap[pix[k]]);
        out.localnoise.push_back(ln);
        out.band.push_back(sl.label);
        out.selfnoise.push_back(in.selfnoise_f64 ? static_cast<float>(in.selfnoise_f64[best_i])
                                                 : in.selfnoise_f32[best_i]);
        out.enf.push_back(enf_at(best_i));
        out.sigma_gmad.push_back(flmax / noise_globmad);

        // adjusted noise from the extreme of the donut box
        float ext_hi = -std::numeric_limits<float>::infinity(), ext_lo = -ext_hi;
        bool has_nan = false;
        for (const float v : sub)
        {
          has_nan |= std::isnan(v);
          ext_hi = std::max(ext_hi, v);
          ext_lo = std::min(ext_lo, v);
        }
        const double extreme = has_nan ? NaN : std::max(std::abs(ext_hi), std::abs(ext_lo));
        const double adjnoise = static_cast<double>(static_cast<float>(extreme)) / ratio;
        const double sig_ext = adjnoise != 0. ? flmax / adjnoise : 0.;
        for (size_t k = start[n]; k < start[n + 1]; ++k)
          snr[pix[k]] = final_f64 ? sig_ext : f32(sig_ext);
        // donut rms and local MAD over its nonzero pixels
        float sig_rms = 0.F, sig_lmad = 0.F;
        if (!fscratch.empty())
        {
          double m = 0.;
          for (const float v : fscratch)
            m += v;
          m /= fscratch.size();
          double ss = 0.;
          for (const float v : fscratch)
            ss += (v - m) * (v - m);
          const float drms = static_cast<float>(std::sqrt(ss / fscratch.size()));
          sig_rms = drms != 0.F ? flmax / drms : 0.F;
          std::vector<float> absd(fscratch.size());
          for (size_t k = 0; k < fscratch.size(); ++k)
            absd[k] = std::abs(fscratch[k]);
          const float lmad = static_cast<float>(np_median(absd)) * 1.4826F;
          sig_lmad = flmax / lmad;
        }
        out.sigma_ext.push_back(sig_ext);
        out.sigma_rms.push_back(sig_rms);
        out.sigma_lmad.push_back(sig_lmad);

        if (cfg.beam_size)
        {
          if (cfg.fixed_beam)
            bk = cfg.fixed_kernel;
          else if (bk.empty())
          {
            float sa = 0.F, sb = 0.F, sp = 0.F;
            for (size_t k = 0; k < bands.size(); ++k)
            {
              sa += in.bmaj[bands[k]] * ww[k];
              sb += in.bmin[bands[k]] * ww[k];
              sp += in.bpa[bands[k]] * ww[k];
            }
            bk = gaussian_kernel(2 * bk_halfsize + 1, sa / sumww, sb / sumww, static_cast<double>(sp / sumww), true);
          }
          // reduced chi2 of the beam fit
          stamp(plane, ny, nx, y, x, bk_halfsize, bstamp);
          const double ssum = pairwise_sum(bstamp.data(), bstamp.size());
          const double ksum = pairwise_sum(bk.data(), bk.size());
          const double scale = ssum / ksum;
          double chi = 0.;
          long cntm = 0;
          for (size_t k = 0; k < bstamp.size(); ++k)
            if (bstamp[k] != 0.F)
            {
              const double d = bstamp[k] - bk[k] * scale;
              chi += d * d;
              ++cntm;
            }
          out.beamchi2.push_back(chi / (adjnoise * adjnoise * static_cast<double>(cntm)));

          // PSF-shape tests
          double smaj, smin, rot;
          if (cfg.fixed_beam)
          {
            smaj = smin = static_cast<double>(2 * bk_halfsize + 1) / 6.0;
            rot = 0.;
          }
          else
          {
            double a = 0, b = 0, c = 0;
            for (size_t k = 0; k < bands.size(); ++k)
            {
              a += bmaj_pl[k] * ww[k];
              b += bmin_pl[k] * ww[k];
              c += bpa_pl[k] * ww[k];
            }
            smaj = a / sumww;
            smin = b / sumww;
            rot = c / sumww;
          }
          const double fw = std::max(2.355 * std::sqrt(std::max(smaj, 1e-3) * std::max(smin, 1e-3)), 1.0);
          const long R6 = std::min(static_cast<long>(std::ceil(6 * fw)), pad - 1);
          const long w6 = 2 * R6 + 1;
          stamp(plane, ny, nx, y, x, R6, pstamp);
          std::vector<double> rr(static_cast<size_t>(w6) * w6);
          std::vector<uint8_t> ann(rr.size());
          double amax = -1.;
          for (long i = 0; i < w6; ++i)
            for (long j = 0; j < w6; ++j)
            {
              const size_t k = static_cast<size_t>(i) * w6 + j;
              rr[k] = std::hypot(static_cast<double>(i - R6), static_cast<double>(j - R6));
              ann[k] = rr[k] >= 2 * fw && rr[k] <= 6 * fw && pstamp[k] != 0.F;
              if (ann[k])
                amax = std::max(amax, static_cast<double>(std::abs(pstamp[k])));
            }
          out.psf_iso.push_back(amax >= 0 ? static_cast<double>(std::abs(flmax) / static_cast<float>(amax)) : NaN);
          // isolation with the other above-threshold islands of the plane left out, dilated by a beam
          std::vector<uint8_t> oth(rr.size(), 0);
          bool any_oth = false;
          for (long i = 0; i < w6; ++i)
          {
            const long yy = y - R6 + i;
            if (yy < 0 || yy >= ny)
              continue;
            for (long j = 0; j < w6; ++j)
            {
              const long xx = x - R6 + j;
              if (xx < 0 || xx >= nx)
                continue;
              const int32_t l = labels.lab[static_cast<size_t>(yy) * nx + xx];
              if (l > 0 && l != n + 1)
              {
                oth[static_cast<size_t>(i) * w6 + j] = 1;
                any_oth = true;
              }
            }
          }
          if (any_oth)
            oth = dilate_l1(oth, w6, std::max(py_round(fw), 1L));
          double amax_x = -1.;
          for (size_t k = 0; k < rr.size(); ++k)
            if (ann[k] && !oth[k])
              amax_x = std::max(amax_x, static_cast<double>(std::abs(pstamp[k])));
          out.psf_iso_x.push_back(amax_x >= 0 ? static_cast<double>(std::abs(flmax) / static_cast<float>(amax_x))
                                              : NaN);
          // correlation of the 1.5-FWHM core with the beam template
          const auto tmpl = gaussian_kernel(w6, smaj, smin, rot, false);
          std::vector<double> dcore, tcore;
          std::vector<float> dcore32;
          const float sg = flmax >= 0 ? 1.F : -1.F;
          for (size_t k = 0; k < rr.size(); ++k)
            if (rr[k] <= 1.5 * fw)
            {
              dcore32.push_back(pstamp[k] * sg);
              dcore.push_back(dcore32.back());
              tcore.push_back(tmpl[k]);
            }
          const double dstd = np_std_f32(dcore32);
          out.psf_corr.push_back(dstd > 0 ? corrcoef(dcore, tcore) : NaN);
        }
        else
        {
          out.beamchi2.push_back(0.);
          out.psf_iso.push_back(NaN);
          out.psf_iso_x.push_back(NaN);
          out.psf_corr.push_back(NaN);
        }
      }
    }
    // procres' finally: a single-band slice is replaced by its SNR image
    if (single)
      for (size_t i = 0; i < npix; ++i)
        single_chan[i] = static_cast<float>(snr[i]);
  }
  out.last_mednoise = mednoise;
  out.any_minsig = minsig_set;
  out.minsig = minsig;
  out.maxsig = maxsig;

  if (nfreq > 1)
  {
    out.has_variance = true;
    out.spvar.assign(npix, 0.F);
    out.spskew.assign(npix, 0.F);
    out.spkurt.assign(npix, 0.F);
    out.counts = 0;
    bool first = true;
    for (long b = 0; b < nfreq; ++b)
    {
      if (!out.usable[b])
        continue;
      ++out.counts;
      const float *p = chan(b);
      for (size_t i = 0; i < npix; ++i)
      {
        const float v = p[i];
        const float v2 = v * v;
        // numpy's float32 power is its own SIMD routine and matches neither x*x*x nor powf bit for bit; the cube
        // and fourth power are taken in double and rounded once (within an ulp of numpy's)
        const double vd = v;
        const float v3 = static_cast<float>(vd * vd * vd), v4 = static_cast<float>((vd * vd) * (vd * vd));
        if (first)
        {
          out.spvar[i] = v2;
          out.spskew[i] = v3;
          out.spkurt[i] = v4;
        }
        else
        {
          out.spvar[i] += v2;
          out.spskew[i] += v3;
          out.spkurt[i] += v4;
        }
      }
      first = false;
    }
    if (in.accumulate)
    {
      // process_plane_result's folding, in one pass under the lock
      const auto &A = *in.accumulate;
      const auto npmax = [](double a, double b) { return (std::isnan(a) || std::isnan(b)) ? NaN : std::max(a, b); };
      std::lock_guard<std::mutex> guard(*A.lock);
      for (size_t i = 0; i < npix; ++i)
      {
        A.spvar_sum[i] += out.spvar[i];
        A.spinvvar_sum[i] += have_inv_var ? inv_var[i] : 0.;
        A.spmodel_sum[i] += have_inv_var ? model_excess[i] : 0.;
        A.spskew_sum[i] += out.spskew[i];
        A.spkurt_sum[i] += out.spkurt[i];
        A.spvar_counts[i] += out.counts;
      }
      for (long s = 0; s < A.nslots; ++s)
      {
        double *vm = A.spvar_max + static_cast<size_t>(s) * npix, *km = A.spkurt_max + static_cast<size_t>(s) * npix;
        for (size_t i = 0; i < npix; ++i)
        {
          vm[i] = npmax(vm[i], out.spvar[i]);
          km[i] = npmax(km[i], out.spkurt[i]);
        }
      }
      for (long b = 0; b < nfreq; ++b)
      {
        if (!out.usable[b])
          continue;
        const float *p = chan(b);
        double *acc = A.pb_var_sum + static_cast<size_t>(b) * npix;
        for (size_t i = 0; i < npix; ++i)
          acc[i] += static_cast<double>(p[i] * p[i]);
        A.pb_var_counts[b] += 1;
      }
      out.has_variance = false;
      out.spvar = {};
      out.spskew = {};
      out.spkurt = {};
    }
    else
    {
      out.pb_var.resize(static_cast<size_t>(nfreq) * npix);
      for (long b = 0; b < nfreq; ++b)
      {
        const float *p = chan(b);
        for (size_t i = 0; i < npix; ++i)
          out.pb_var[static_cast<size_t>(b) * npix + i] = p[i] * p[i];
      }
      if (have_inv_var)
      {
        out.inv_var = std::move(inv_var);
        out.model_excess = std::move(model_excess);
      }
    }
  }
}

} // namespace breifast::spectron
