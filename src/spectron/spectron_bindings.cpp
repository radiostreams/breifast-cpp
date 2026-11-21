// pybind11 binding of spectron::process_plane: zero-copy views of procres' numpy arrays, GIL released for the plane.
#include "spectron_bindings.hpp"
#include "spectron_plane.hpp"

#include <algorithm>
#include <map>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace breifast::spectron
{
namespace
{
std::mutex pb_ivar_mutex;
std::mutex accumulate_mutex;

template <typename T> bool is_dtype(const py::array &a)
{
  return a.dtype().equal(py::dtype::of<T>());
}

std::string where(const char *name)
{
  return std::string("process_plane_spectron: ") + name;
}

/// Checks that the last two axes of `a` are (ny, nx).
void check_shape(const py::array &a, long ny, long nx, const char *name)
{
  const auto nd = a.ndim();
  if (nd < 2 || a.shape(nd - 2) != ny || a.shape(nd - 1) != nx)
    throw std::invalid_argument(where(name) + ": planes must be (ny, nx)");
}

/// Whether the last two axes of `a` are a C-contiguous (ny, nx) plane.
bool contiguous_planes(const py::array &a)
{
  const auto nd = a.ndim();
  return a.strides(nd - 1) == static_cast<long>(a.itemsize()) &&
         a.strides(nd - 2) == a.shape(nd - 1) * static_cast<long>(a.itemsize());
}

/// Checks that the last two axes of `a` are a C-contiguous (ny, nx) plane: for caller-owned arrays the kernel
/// writes into (the accumulators), which procres allocates in C order.
void check_planes(const py::array &a, long ny, long nx, const char *name)
{
  check_shape(a, ny, nx, name);
  if (!contiguous_planes(a))
    throw std::invalid_argument(where(name) + ": planes must be C-contiguous");
}

/// Copies `nb` strided (ny, nx) planes into C-contiguous `dst` (nb * ny * nx), in 64 x 64 tiles so that a
/// transposed source -- a store kept in (l, m) order read in (m, l) order -- is read cache-friendly.
/// Strides in bytes.
template <typename T>
void gather_planes(const char *src, long nb, long sb, long sy, long sx, long ny, long nx, T *dst)
{
  constexpr long B = 64;
  for (long b = 0; b < nb; ++b)
    for (long y0 = 0; y0 < ny; y0 += B)
      for (long x0 = 0; x0 < nx; x0 += B)
        for (long y = y0; y < std::min(y0 + B, ny); ++y)
        {
          const char *row = src + b * sb + y * sy;
          T *out = dst + (static_cast<size_t>(b) * ny + y) * nx;
          for (long x = x0; x < std::min(x0 + B, nx); ++x)
            out[x] = *reinterpret_cast<const T *>(row + x * sx);
        }
}

/// The reverse of gather_planes: C-contiguous `src` back into strided planes.
template <typename T>
void scatter_planes(const T *src, long nb, char *dst, long sb, long sy, long sx, long ny, long nx)
{
  constexpr long B = 64;
  for (long b = 0; b < nb; ++b)
    for (long y0 = 0; y0 < ny; y0 += B)
      for (long x0 = 0; x0 < nx; x0 += B)
        for (long y = y0; y < std::min(y0 + B, ny); ++y)
        {
          char *row = dst + b * sb + y * sy;
          const T *in = src + (static_cast<size_t>(b) * ny + y) * nx;
          for (long x = x0; x < std::min(x0 + B, nx); ++x)
            *reinterpret_cast<T *>(row + x * sx) = in[x];
        }
}
} // namespace

/// C-contiguous copies of per-run input images that came with planes in another memory order (the current
/// forward-difference and convolution stores keep (l, m) last; procres reads them by name as (m, l), which
/// gives a transposed view). Keyed by buffer address, shape and strides; each entry holds a reference to its
/// source array, so the address cannot be reused by another array while the run lasts.
struct InputCache
{
  struct Entry
  {
    py::object source;
    std::vector<char> data;
  };
  std::mutex lock;
  std::map<std::string, std::shared_ptr<Entry>> entries;
};

namespace
{
/// The C-contiguous data of a per-run (nb, ny, nx) or (ny, nx) image: its own buffer when its planes are
/// C-contiguous, else a copy made once per run and kept on the Config. Sets `band_stride` (elements; 0 for a
/// single or broadcast band).
template <typename T> const T *run_image(const Config &config, const py::array &a, long &band_stride)
{
  const long nd = a.ndim();
  const long nb = nd == 3 ? a.shape(0) : 1;
  const bool broadcast = nd == 3 && (nb == 1 || a.strides(0) == 0);
  if (contiguous_planes(a))
  {
    band_stride = broadcast || nd == 2 ? 0 : static_cast<long>(a.strides(0) / a.itemsize());
    return static_cast<const T *>(a.data());
  }
  std::string key = std::to_string(reinterpret_cast<uintptr_t>(a.data())) + ":" + std::string(1, a.dtype().kind()) + std::to_string(a.itemsize());
  for (long i = 0; i < nd; ++i)
    key += ":" + std::to_string(a.shape(i)) + "/" + std::to_string(a.strides(i));
  if (!config.input_cache)
    config.input_cache = std::make_shared<InputCache>();
  auto &cache = *config.input_cache;
  std::lock_guard<std::mutex> guard(cache.lock);
  auto &entry = cache.entries[key];
  const long ny = a.shape(nd - 2), nx = a.shape(nd - 1);
  const long ncopy = broadcast ? 1 : nb;
  if (!entry)
  {
    entry = std::make_shared<InputCache::Entry>();
    entry->source = a;
    entry->data.resize(static_cast<size_t>(ncopy) * ny * nx * sizeof(T));
    gather_planes<T>(static_cast<const char *>(a.data()), ncopy, nd == 3 ? a.strides(0) : 0, a.strides(nd - 2),
                     a.strides(nd - 1), ny, nx, reinterpret_cast<T *>(entry->data.data()));
  }
  band_stride = ncopy > 1 ? ny * nx : 0;
  return reinterpret_cast<const T *>(entry->data.data());
}

template <typename T> const T *plane2(const Config &config, const py::object &o, long ny, long nx, const char *name)
{
  if (o.is_none())
    return nullptr;
  py::array a = o;
  if (a.ndim() != 2 || !is_dtype<T>(a))
    throw std::invalid_argument(where(name) + ": expected a 2D array of the right dtype");
  check_shape(a, ny, nx, name);
  long unused;
  return run_image<T>(config, a, unused);
}

/// A 2D float32 or float64 array: exactly one of the two pointers is set (or neither, for None).
void plane2f(const Config &config, const py::object &o, long ny, long nx, const char *name, const float *&p32,
             const double *&p64)
{
  p32 = nullptr;
  p64 = nullptr;
  if (o.is_none())
    return;
  py::array a = o;
  if (is_dtype<float>(a))
    p32 = plane2<float>(config, o, ny, nx, name);
  else if (is_dtype<double>(a))
    p64 = plane2<double>(config, o, ny, nx, name);
  else
    throw std::invalid_argument(where(name) + ": expected float32 or float64");
}

template <typename T>
BandArray<T> band3(const Config &config, const py::object &o, long nfreq, long ny, long nx, const char *name)
{
  BandArray<T> b;
  if (o.is_none())
    return b;
  py::array a = o;
  if (a.ndim() != 3 || !is_dtype<T>(a) || (a.shape(0) != nfreq && a.shape(0) != 1))
    throw std::invalid_argument(where(name) + ": expected a (nfreq, ny, nx) array of the right dtype");
  check_shape(a, ny, nx, name);
  b.data = run_image<T>(config, a, b.band_stride);
  return b;
}

BandArrayF band3f(const Config &config, const py::object &o, long nfreq, long ny, long nx, const char *name)
{
  BandArrayF b;
  if (o.is_none())
    return b;
  py::array a = o;
  if (is_dtype<float>(a))
    b.f32 = band3<float>(config, o, nfreq, ny, nx, name);
  else if (is_dtype<double>(a))
    b.f64 = band3<double>(config, o, nfreq, ny, nx, name);
  else
    throw std::invalid_argument(where(name) + ": expected float32 or float64");
  return b;
}

template <typename T> std::vector<T> vec1(const py::object &o, long n, const char *name)
{
  if (o.is_none())
    return {};
  auto a = py::array_t<T, py::array::forcecast>::ensure(o);
  if (!a || a.ndim() != 1 || a.shape(0) != n)
    throw std::invalid_argument(where(name) + ": expected a 1D array of length nfreq");
  std::vector<T> v(n);
  for (long i = 0; i < n; ++i)
    v[i] = a.at(i);
  return v;
}

/// Hand a vector over to numpy without a copy.
template <typename T> py::array to_numpy(std::vector<T> &&v, std::vector<py::ssize_t> shape)
{
  auto *heap = new std::vector<T>(std::move(v));
  py::capsule owner(heap, [](void *p) { delete static_cast<std::vector<T> *>(p); });
  return py::array_t<T>(shape, heap->data(), owner);
}

py::dict process_plane_spectron(const Config &config, py::array cube, long stokes_index, py::object u_index,
                                py::object nonzero_gate, py::object weights, py::object chanwidth,
                                py::object accept_mask, py::object mean_image, py::object snr_cutoff,
                                py::object flux_cutoff, py::object selfnoise_offset, py::object dde_noise,
                                py::object nm_noise, py::object nm_noise_band, py::object noise_scale_sqrt,
                                py::object selfnoise, py::object baseline_flux, py::object bmaj, py::object bmin,
                                py::object bpa, std::vector<double> slice_threshold, py::object pb_ivar,
                                py::object accumulators)
{
  const long ny = config.ny, nx = config.nx, nfreq = config.nfreq;
  if (!is_dtype<float>(cube) || cube.ndim() != 4 || cube.shape(1) != nfreq || !cube.writeable())
    throw std::invalid_argument(where("cube") + ": expected a writeable float32 (stokes, nfreq, ny, nx) array");
  check_shape(cube, ny, nx, "cube");
  // planes in another memory order (a store kept in (l, m) order, read as (m, l)) are copied into C-contiguous
  // buffers for the plane, and the slot the kernel changes is written back after it
  const bool cube_strided = !contiguous_planes(cube);
  if (slice_threshold.size() != config.band_slices.size())
    throw std::invalid_argument(where("slice_threshold") + ": one threshold per band slice");

  PlaneInputs in;
  auto *base = static_cast<char *>(cube.mutable_data());
  char *slot = base + stokes_index * cube.strides(0);
  char *slot_u = u_index.is_none() ? nullptr : base + u_index.cast<long>() * cube.strides(0);
  const long sb = cube.strides(1), sy = cube.strides(2), sx = cube.strides(3);
  thread_local std::vector<float> slot_buf, slot_u_buf;
  if (!cube_strided)
  {
    in.cube = reinterpret_cast<float *>(slot);
    in.cube_band_stride = static_cast<long>(sb / 4);
    if (slot_u)
    {
      in.cube_u = reinterpret_cast<float *>(slot_u);
      in.cube_u_band_stride = in.cube_band_stride;
    }
  }
  in.nonzero_gate = vec1<uint8_t>(nonzero_gate, nfreq, "nonzero_gate");
  in.weights = vec1<float>(weights, nfreq, "weights");
  in.chanwidth = vec1<float>(chanwidth, nfreq, "chanwidth");
  in.accept_mask = band3<uint8_t>(config, py::array(accept_mask).view("uint8"), nfreq, ny, nx, "accept_mask");
  in.mean_image = band3<float>(config, mean_image, nfreq, ny, nx, "mean_image");
  if (config.subtract_mean && !in.mean_image)
    throw std::invalid_argument(where("mean_image") + ": needed with subtract_mean");
  in.snr_cutoff = band3f(config, snr_cutoff, nfreq, ny, nx, "snr_cutoff");
  in.flux_cutoff = band3f(config, flux_cutoff, nfreq, ny, nx, "flux_cutoff");
  plane2f(config, selfnoise_offset, ny, nx, "selfnoise_offset", in.selfnoise_offset_f32, in.selfnoise_offset_f64);
  plane2f(config, dde_noise, ny, nx, "dde_noise", in.dde_noise_f32, in.dde_noise_f64);
  plane2f(config, nm_noise, ny, nx, "nm_noise", in.nm_noise_f32, in.nm_noise_f64);
  plane2f(config, selfnoise, ny, nx, "selfnoise", in.selfnoise_f32, in.selfnoise_f64);
  if ((!in.selfnoise_offset_f32 && !in.selfnoise_offset_f64) || (!in.dde_noise_f32 && !in.dde_noise_f64) ||
      (!in.nm_noise_f32 && !in.nm_noise_f64) || (!in.selfnoise_f32 && !in.selfnoise_f64))
    throw std::invalid_argument(where("noise terms") + ": selfnoise_offset, dde_noise, nm_noise and selfnoise are required");
  in.nm_noise_band = band3<double>(config, nm_noise_band, nfreq, ny, nx, "nm_noise_band");
  in.noise_scale_sqrt = plane2<double>(config, noise_scale_sqrt, ny, nx, "noise_scale_sqrt");
  in.baseline_flux = plane2<float>(config, baseline_flux, ny, nx, "baseline_flux");
  if (config.beam_size && !config.fixed_beam)
  {
    in.bmaj = vec1<float>(bmaj, nfreq, "bmaj");
    in.bmin = vec1<float>(bmin, nfreq, "bmin");
    in.bpa = vec1<float>(bpa, nfreq, "bpa");
    if (in.bmaj.empty() || in.bmin.empty() || in.bpa.empty())
      throw std::invalid_argument(where("beam") + ": per-band bmaj/bmin/bpa needed for the per-plane beam");
  }
  in.slice_threshold = std::move(slice_threshold);
  if (!pb_ivar.is_none())
  {
    py::array a = pb_ivar;
    if (!is_dtype<double>(a) || a.ndim() != 3 || a.shape(0) != nfreq || !a.writeable() ||
        a.strides(0) != static_cast<long>(ny * nx * 8))
      throw std::invalid_argument(where("pb_ivar") + ": expected a writeable contiguous float64 (nfreq, ny, nx)");
    check_planes(a, ny, nx, "pb_ivar");
    in.pb_ivar = static_cast<double *>(a.mutable_data());
    in.pb_ivar_lock = &pb_ivar_mutex;
  }

  if (!accumulators.is_none())
  {
    // procres' variance accumulators, as a dict of the arrays process_plane_result folds into
    const py::dict d = accumulators;
    const auto f64 = [&](const char *key, long lead) -> py::array
    {
      py::array a = d[key];
      if (!is_dtype<double>(a) || !a.writeable() || a.ndim() != (lead ? 3 : 2) || (lead && a.shape(0) != lead) ||
          a.strides(0) != static_cast<long>((lead ? ny * nx : nx) * 8))
        throw std::invalid_argument(where(key) + ": expected a writeable contiguous float64 array");
      check_planes(a, ny, nx, key);
      return a;
    };
    PlaneInputs::Accumulators A;
    A.spvar_sum = static_cast<double *>(f64("spvar_sum", 0).mutable_data());
    A.spskew_sum = static_cast<double *>(f64("spskew_sum", 0).mutable_data());
    A.spkurt_sum = static_cast<double *>(f64("spkurt_sum", 0).mutable_data());
    A.spinvvar_sum = static_cast<double *>(f64("spinvvar_sum", 0).mutable_data());
    A.spmodel_sum = static_cast<double *>(f64("spmodel_sum", 0).mutable_data());
    py::array vmax = d["spvar_max"];
    A.nslots = vmax.ndim() == 3 ? static_cast<long>(vmax.shape(0)) : 0;
    A.spvar_max = static_cast<double *>(f64("spvar_max", A.nslots).mutable_data());
    A.spkurt_max = static_cast<double *>(f64("spkurt_max", A.nslots).mutable_data());
    A.pb_var_sum = static_cast<double *>(f64("pb_var_sum", nfreq).mutable_data());
    py::array cnt = d["spvar_counts"], pbc = d["pb_var_counts"];
    if (!is_dtype<int64_t>(cnt) || cnt.ndim() != 2 || !cnt.writeable() || !is_dtype<int64_t>(pbc) ||
        pbc.ndim() != 1 || pbc.shape(0) != nfreq || !pbc.writeable() || pbc.strides(0) != 8)
      throw std::invalid_argument(where("counts") + ": expected writeable int64 spvar_counts (ny, nx), pb_var_counts (nfreq)");
    check_planes(cnt, ny, nx, "spvar_counts");
    A.spvar_counts = static_cast<int64_t *>(cnt.mutable_data());
    A.pb_var_counts = static_cast<int64_t *>(pbc.mutable_data());
    A.lock = &accumulate_mutex;
    in.accumulate = A;
  }

  PlaneResult out;
  {
    py::gil_scoped_release release;
    if (cube_strided)
    {
      const size_t n = static_cast<size_t>(nfreq) * ny * nx;
      slot_buf.resize(n);
      gather_planes<float>(slot, nfreq, sb, sy, sx, ny, nx, slot_buf.data());
      in.cube = slot_buf.data();
      in.cube_band_stride = ny * nx;
      if (slot_u)
      {
        slot_u_buf.resize(n);
        gather_planes<float>(slot_u, nfreq, sb, sy, sx, ny, nx, slot_u_buf.data());
        in.cube_u = slot_u_buf.data();
        in.cube_u_band_stride = ny * nx;
      }
    }
    process_plane(config, in, out);
    if (cube_strided) // the kernel changes only the detection slot (P formed in it, single-band SNR planes)
      scatter_planes<float>(slot_buf.data(), nfreq, slot, sb, sy, sx, ny, nx);
  }

  py::dict r;
  r["usable"] = to_numpy(std::move(out.usable), {nfreq}).attr("astype")("bool");
  r["live_without_finite"] = out.live_without_finite;
  r["returned_early"] = out.returned_early;
  r["minsig"] = out.any_minsig ? py::object(py::float_(out.minsig)) : py::none();
  r["maxsig"] = out.any_minsig ? py::object(py::float_(out.maxsig)) : py::none();
  py::list hists;
  for (auto &h : out.hist)
    hists.append(h ? py::object(to_numpy(std::move(*h), {static_cast<py::ssize_t>(2 * config.max_sigma_bin)}))
                   : py::none());
  r["hist"] = py::tuple(hists);
  r["mednoises"] = to_numpy(std::move(out.mednoises), {static_cast<py::ssize_t>(config.band_slices.size())});
  r["last_mednoise"] = out.last_mednoise;

  const py::ssize_t nd = static_cast<py::ssize_t>(out.x.size());
  py::dict det;
  det["x"] = to_numpy(std::move(out.x), {nd});
  det["y"] = to_numpy(std::move(out.y), {nd});
  det["band"] = out.band;
  det["peak_flux"] = to_numpy(std::move(out.flux), {nd});
  det["sigma"] = to_numpy(std::move(out.sigma), {nd});
  det["sigma_ext"] = to_numpy(std::move(out.sigma_ext), {nd});
  det["sigma_rms"] = to_numpy(std::move(out.sigma_rms), {nd});
  det["flooded"] = to_numpy(std::move(out.flooded), {nd});
  det["sigma_lmad"] = to_numpy(std::move(out.sigma_lmad), {nd});
  det["sigma_gmad"] = to_numpy(std::move(out.sigma_gmad), {nd});
  det["selfnoise"] = to_numpy(std::move(out.selfnoise), {nd});
  det["enf"] = to_numpy(std::move(out.enf), {nd});
  det["beamchi2"] = to_numpy(std::move(out.beamchi2), {nd});
  det["psf_iso"] = to_numpy(std::move(out.psf_iso), {nd});
  det["psf_iso_x"] = to_numpy(std::move(out.psf_iso_x), {nd});
  det["psf_corr"] = to_numpy(std::move(out.psf_corr), {nd});
  det["dip_src"] = to_numpy(std::move(out.dip_src), {nd});
  det["dip_mirror"] = to_numpy(std::move(out.dip_mirror), {nd});
  det["dip_sep"] = to_numpy(std::move(out.dip_sep), {nd});
  det["boost"] = to_numpy(std::move(out.boost), {nd});
  det["size"] = to_numpy(std::move(out.size), {nd});
  det["ln"] = to_numpy(std::move(out.localnoise), {nd});
  r["detections"] = det;

  py::dict peaks;
  for (auto &[label, values] : out.nominal_peaks)
  {
    const py::ssize_t n = static_cast<py::ssize_t>(values.size() / 3);
    peaks[py::str(label)] = to_numpy(std::move(values), {n, 3});
  }
  r["nominal_peaks"] = peaks;
  r["flood_dropped_islands"] = out.flood_dropped_islands;
  r["flood_dropped_cells"] = out.flood_dropped_cells;
  r["edge_dropped"] = out.edge_dropped;
  r["rms_dropped"] = out.rms_dropped;
  r["flood_tagged"] = out.flood_tagged;

  if (out.has_variance)
  {
    py::dict v;
    v["spvar"] = to_numpy(std::move(out.spvar), {ny, nx});
    v["spskew"] = to_numpy(std::move(out.spskew), {ny, nx});
    v["spkurt"] = to_numpy(std::move(out.spkurt), {ny, nx});
    v["counts"] = out.counts;
    v["pb_var"] = to_numpy(std::move(out.pb_var), {nfreq, ny, nx});
    v["inv_var"] = out.inv_var.empty() ? py::object(py::float_(0.)) : py::object(to_numpy(std::move(out.inv_var), {ny, nx}));
    v["model_excess"] = out.model_excess.empty() ? py::object(py::float_(0.))
                                                 : py::object(to_numpy(std::move(out.model_excess), {ny, nx}));
    r["variance"] = v;
  }
  else
    r["variance"] = py::none();
  return r;
}
} // namespace
} // namespace breifast::spectron

void init_spectron_bindings(py::module_ &m)
{
  using breifast::spectron::Config;
  py::class_<Config> cfg(m, "SpectronConfig", "Per-run settings of the spectron process_plane port");
  cfg.def(py::init<>())
      .def_readwrite("ny", &Config::ny)
      .def_readwrite("nx", &Config::nx)
      .def_readwrite("nfreq", &Config::nfreq)
      .def(
          "set_band_slices",
          [](Config &c, const std::vector<std::tuple<long, long, std::string>> &slices)
          {
            c.band_slices.clear();
            for (const auto &[a, b, l] : slices)
              c.band_slices.push_back({a, b, l});
          },
          "Band slices as (start, stop, label), in procres' (forward) order")
      .def_readwrite("subtract_mean", &Config::subtract_mean)
      .def_property(
          "noise_estimator",
          [](const Config &c) { return c.noise_estimator == Config::NoiseEstimator::rms ? "rms" : "mad"; },
          [](Config &c, const std::string &s)
          {
            if (s == "rms")
              c.noise_estimator = Config::NoiseEstimator::rms;
            else if (s == "thermal")
              throw std::invalid_argument("the thermal noise estimator is not ported");
            else
              c.noise_estimator = Config::NoiseEstimator::mad;
          })
      .def_readwrite("boxsize", &Config::boxsize)
      .def_readwrite("inner_boxsize", &Config::inner_boxsize)
      .def_readwrite("noise_floor_excess", &Config::noise_floor_excess)
      .def_readwrite("selfnoise_bandwidth_scaling", &Config::selfnoise_bandwidth_scaling)
      .def_readwrite("ddenoise_scaling_factor", &Config::ddenoise_scaling_factor)
      .def_readwrite("full_bandwidth", &Config::full_bandwidth)
      .def_readwrite("reject_noise_threshold", &Config::reject_noise_threshold)
      .def_readwrite("histogram_detail", &Config::histogram_detail)
      .def_readwrite("max_sigma_bin", &Config::max_sigma_bin)
      .def_readwrite("skip_detection", &Config::skip_detection)
      .def_readwrite("nominal_peaks", &Config::nominal_peaks)
      .def_readwrite("threshold", &Config::threshold)
      .def_readwrite("flood_cell_max", &Config::flood_cell_max)
      .def_readwrite("flood_cell_size", &Config::flood_cell_size)
      .def_readwrite("flood_cell_hard", &Config::flood_cell_hard)
      .def_readwrite("edge_margin", &Config::edge_margin)
      .def_readwrite("min_sigma_rms", &Config::min_sigma_rms)
      .def_readwrite("beam_size", &Config::beam_size)
      .def_readwrite("fixed_beam", &Config::fixed_beam)
      .def_readwrite("fixed_kernel", &Config::fixed_kernel)
      .def_readwrite("fixed_halfsize", &Config::fixed_halfsize)
      .def_readwrite("fixed_bmaj", &Config::fixed_bmaj)
      .def_readwrite("dip_x", &Config::dip_x)
      .def_readwrite("dip_y", &Config::dip_y)
      .def_readwrite("dip_flux", &Config::dip_flux)
      .def_readwrite("dipole_radius", &Config::dipole_radius);

  m.def("process_plane_spectron", &breifast::spectron::process_plane_spectron, py::arg("config"), py::arg("cube"),
        py::arg("stokes_index"), py::arg("u_index"), py::arg("nonzero_gate"), py::arg("weights"),
        py::arg("chanwidth"), py::arg("accept_mask"), py::arg("mean_image"), py::arg("snr_cutoff"),
        py::arg("flux_cutoff"), py::arg("selfnoise_offset"), py::arg("dde_noise"), py::arg("nm_noise"),
        py::arg("nm_noise_band"), py::arg("noise_scale_sqrt"), py::arg("selfnoise"), py::arg("baseline_flux"),
        py::arg("bmaj"), py::arg("bmin"), py::arg("bpa"), py::arg("slice_threshold"), py::arg("pb_ivar"),
        py::arg("accumulators") = py::none(),
        "spectron's process_plane for one (interval, Stokes) plane; modifies `cube` in place as procres does");
  m.def("max_to_std_ratio", &breifast::spectron::max_to_std_ratio);
  m.attr("SPECTRON_KERNEL_VERSION") = 6;
}
