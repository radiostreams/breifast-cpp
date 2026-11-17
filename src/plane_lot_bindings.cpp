#include "plane_lot_bindings.hpp"
#include "plane_lot_dispatch.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "host_copies.hpp"

#ifdef BREIFAST_HAVE_GPU
#include "gpu/gpu_device.hpp"
#include "gpu/gpu_plane.hpp"
#endif

#define XTENSOR_USE_TBB
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

namespace breifast
{
/// Arrays that must outlive the registered copies: an input that is not C-contiguous or not of the expected dtype is
/// first converted by numpy, into a temporary that lives here.
using KeepAlive = std::vector<pybind11::array>;

/// An xtensor with the shape of `object`, allocated but not filled: the copy of its elements is registered in `copies`,
/// to run later in parallel without the GIL. Moving the result keeps its buffer, which the copy targets.
template <typename T, size_t N, typename Tensor = xt::xtensor<T, N>>
Tensor to_xtensor(const pybind11::object &object, const char *key, HostCopies &copies, KeepAlive &keep_alive)
{
  auto array = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>::ensure(object);
  if (!array)
  {
    throw std::invalid_argument(std::string("'") + key + "' is not an array of the expected type");
  }
  if (static_cast<size_t>(array.ndim()) != N)
  {
    throw std::invalid_argument(std::string("'") + key + "' has " + std::to_string(array.ndim()) +
                                " dimensions, expected " + std::to_string(N));
  }
  std::array<size_t, N> shape{};
  for (size_t i = 0; i < N; ++i)
  {
    shape[i] = static_cast<size_t>(array.shape(static_cast<pybind11::ssize_t>(i)));
  }
  Tensor tensor(shape);
  copies.add(array.data(), tensor.data(), static_cast<size_t>(array.nbytes()));
  keep_alive.push_back(std::move(array));
  return tensor;
}

template <typename T, size_t N, typename Tensor = xt::xtensor<T, N>>
Tensor require_xtensor(const pybind11::dict &d, const char *key, HostCopies &copies, KeepAlive &keep_alive)
{
  return to_xtensor<T, N, Tensor>(d[key], key, copies, keep_alive);
}

template <typename T, size_t N>
std::optional<xt::xtensor<T, N>> optional_xtensor_key(const pybind11::dict &d, const char *key, HostCopies &copies,
                                                      KeepAlive &keep_alive)
{
  if (!d.contains(key))
  {
    return std::nullopt;
  }
  return to_xtensor<T, N>(d[key], key, copies, keep_alive);
}

BandSlice to_band_slice(const pybind11::dict &d)
{
  BandSlice band;
  band.start = d["start"].cast<long>();
  band.stop = d["stop"].cast<long>();
  band.selfnoise_scale_factor = d["selfnoise_scale_factor"].cast<float>();
  band.label = d["label"].cast<std::string>();
  return band;
}

PlaneBatchItem to_plane_batch_item(const pybind11::dict &plane_dict, HostCopies &copies, KeepAlive &keep_alive)
{
  PlaneBatchItem item;
  item.stokes_cube = require_xtensor<float, 3, HostCube>(plane_dict, "stokes_cube", copies, keep_alive);
  item.weights = require_xtensor<float, 1>(plane_dict, "weights", copies, keep_alive);
  item.nonzero_channels = require_xtensor<bool, 1>(plane_dict, "nonzero_channels", copies, keep_alive);
  item.group_index = plane_dict["group_index"].cast<size_t>();
  item.stokes_label = plane_dict["stokes_label"].cast<std::string>();
  item.beam_major_axis = optional_xtensor_key<float, 1>(plane_dict, "beam_major_axis", copies, keep_alive);
  item.beam_minor_axis = optional_xtensor_key<float, 1>(plane_dict, "beam_minor_axis", copies, keep_alive);
  item.beam_position_angle = optional_xtensor_key<float, 1>(plane_dict, "beam_position_angle", copies, keep_alive);
  return item;
}

PlaneBatchGroup to_plane_batch_group(const pybind11::dict &group_dict, HostCopies &copies, KeepAlive &keep_alive)
{
  PlaneBatchGroup group;
  group.accept_mask = require_xtensor<bool, 3>(group_dict, "accept_mask", copies, keep_alive);
  group.mean_image = optional_xtensor_key<float, 3>(group_dict, "mean_image", copies, keep_alive);
  group.selfnoise_offset = require_xtensor<float, 2>(group_dict, "selfnoise_offset", copies, keep_alive);
  group.selfnoise_raw = require_xtensor<float, 2>(group_dict, "selfnoise_raw", copies, keep_alive);
  group.dde_noise = require_xtensor<float, 2>(group_dict, "dde_noise", copies, keep_alive);
  group.flux_cutoff = optional_xtensor_key<float, 3>(group_dict, "flux_cutoff", copies, keep_alive);
  group.snr_cutoff = optional_xtensor_key<float, 3>(group_dict, "snr_cutoff", copies, keep_alive);
  return group;
}

template <typename T, size_t N>
std::optional<xt::xtensor<T, N>> to_optional_xtensor(const pybind11::object &obj, const char *key, HostCopies &copies,
                                                     KeepAlive &keep_alive)
{
  if (obj.is_none())
  {
    return std::nullopt;
  }
  return to_xtensor<T, N>(obj, key, copies, keep_alive);
}

pybind11::object to_python_result(const ProcessPlaneResult &result)
{
  pybind11::dict out;
  out["any_data"] = result.any_data;
  out["min_sigma"] = result.min_sigma;
  out["max_sigma"] = result.max_sigma;
  out["histogram0"] = result.histograms[0];
  out["histogram1"] = result.histograms[1];
  out["histogram2"] = result.histograms[2];
  out["median_noises"] = result.median_noises;

  pybind11::dict detections;
  detections["peak_x"] = result.detections.peak_x();
  detections["peak_y"] = result.detections.peak_y();
  detections["island_size"] = result.detections.island_size();
  detections["stokes_label"] = result.detections.stokes_label();
  detections["band_label"] = result.detections.band_label();
  detections["flux"] = result.detections.flux();
  detections["local_noise"] = result.detections.local_noise();
  detections["local_noise_median_ratio"] = result.detections.local_noise_median_ratio();
  detections["signal_to_noise"] = result.detections.signal_to_noise();
  detections["signal_to_noise_rms"] = result.detections.signal_to_noise_rms();
  detections["signal_to_noise_extended_box"] = result.detections.signal_to_noise_extended_box();
  detections["signal_to_noise_local_mad"] = result.detections.signal_to_noise_local_mad();
  detections["signal_to_noise_global_mad"] = result.detections.signal_to_noise_global_mad();
  detections["self_noise"] = result.detections.self_noise();
  detections["excess_noise_factor"] = result.detections.excess_noise_factor();
  detections["beam_chi2"] = result.detections.beam_chi2();
  detections["boost"] = result.detections.boost();
  out["detections"] = detections;

  if (result.variance_stats)
  {
    const auto &vs = *result.variance_stats;
    pybind11::dict variance_stats;
    variance_stats["sum_pow2"] = xt::pyarray<float>(vs.sum_pow2);
    variance_stats["sum_pow3"] = xt::pyarray<float>(vs.sum_pow3);
    variance_stats["sum_pow4"] = xt::pyarray<float>(vs.sum_pow4);
    variance_stats["nonzero_channels_mask"] = xt::pyarray<bool>(vs.nonzero_channels_mask);
    if (vs.per_channels_squared)
    {
      variance_stats["per_channels_squared"] = xt::pyarray<float>(*vs.per_channels_squared);
    }
    out["variance_stats"] = variance_stats;
  }
  else
  {
    out["variance_stats"] = pybind11::none();
  }

  return out;
}

// Replaces breifast.py:1173-1192 (minus the prefetch, see plane_lot_dispatch.hpp's header comment):
// `planes` is one entry per (interval, stokes) pair in the lot, `stokes_groups` is the shared
// per-Stokes arrays those planes reference by index -- same two-level shape as process_plane_batch's
// existing `stokes_groups` contract, just with the per-plane items pulled out to their own list.
pybind11::list process_lot(pybind11::list planes, pybind11::list stokes_groups, pybind11::list band_slices,
                           pybind11::object baseline_flux_obj, bool subtract_mean, bool use_rms_noise_estimator,
                           int boxsize, int inner_boxsize, float ddenoise_scaling_factor, float full_bandwidth,
                           float reject_noise_threshold, int histogram_detail, int max_sigma_bin, float threshold,
                           bool save_stat_cubes, bool beam_size_enabled, pybind11::object fixed_beam_kernel_obj,
                           int fixed_beam_halfsize, int num_threads, bool use_gpu)
{
  // The arrays are copied into the lot in parallel, after the GIL is released: the conversions below only allocate and
  // register the copies. The vectors are reserved and their elements nothrow-movable, so that the buffers the copies
  // target never move.
  static_assert(std::is_nothrow_move_constructible_v<PlaneBatchItem> &&
                std::is_nothrow_move_constructible_v<PlaneBatchGroup>);
  HostCopies copies;
  KeepAlive keep_alive;
  LotBatch lot;
  lot.planes.reserve(planes.size());
  lot.groups.reserve(stokes_groups.size());
  for (auto plane_handle : planes)
  {
    lot.planes.push_back(to_plane_batch_item(plane_handle.cast<pybind11::dict>(), copies, keep_alive));
  }
  for (auto group_handle : stokes_groups)
  {
    lot.groups.push_back(to_plane_batch_group(group_handle.cast<pybind11::dict>(), copies, keep_alive));
  }

  PlaneConfig config;
  config.subtract_mean = subtract_mean;
  config.use_rms_noise_estimator = use_rms_noise_estimator;
  config.boxsize = boxsize;
  config.inner_boxsize = inner_boxsize;
  config.ddenoise_scaling_factor = ddenoise_scaling_factor;
  config.full_bandwidth = full_bandwidth;
  config.reject_noise_threshold = reject_noise_threshold;
  config.histogram_detail = histogram_detail;
  config.max_sigma_bin = max_sigma_bin;
  config.threshold = threshold;
  config.save_stat_cubes = save_stat_cubes;
  config.beam_size_enabled = beam_size_enabled;
  config.fixed_beam_kernel =
      to_optional_xtensor<double, 2>(fixed_beam_kernel_obj, "fixed_beam_kernel", copies, keep_alive);
  config.fixed_beam_halfsize = fixed_beam_halfsize;
  config.baseline_flux = to_optional_xtensor<float, 2>(baseline_flux_obj, "baseline_flux", copies, keep_alive);
  for (auto band_slice_handle : band_slices)
  {
    config.band_slices.push_back(to_band_slice(band_slice_handle.cast<pybind11::dict>()));
  }

#ifndef BREIFAST_HAVE_GPU
  if (use_gpu)
  {
    throw std::runtime_error("breifast_helpers built without GPU support");
  }
#endif

  std::vector<ProcessPlaneResult> results;
  {
    pybind11::gil_scoped_release release;
    copies.run();
#ifdef BREIFAST_HAVE_GPU
    if (use_gpu)
    {
      results = gpu::run_batch_core_gpu(lot.planes, lot.groups, config);
    }
    else
    {
      results = run_lot_core(lot, config, num_threads);
    }
#else
    results = run_lot_core(lot, config, num_threads);
#endif
  }

  pybind11::list out;
  for (const ProcessPlaneResult &result : results)
  {
    out.append(to_python_result(result));
  }
  return out;
}
} // namespace breifast

void init_plane_lot_bindings(pybind11::module_ &m)
{
  m.def("process_lot", &breifast::process_lot, pybind11::arg("planes"), pybind11::arg("stokes_groups"),
        pybind11::arg("band_slices"), pybind11::arg("baseline_flux") = pybind11::none(), pybind11::arg("subtract_mean"),
        pybind11::arg("use_rms_noise_estimator") = true, pybind11::arg("boxsize"), pybind11::arg("inner_boxsize") = 10,
        pybind11::arg("ddenoise_scaling_factor") = 1.0f, pybind11::arg("full_bandwidth"),
        pybind11::arg("reject_noise_threshold") = 0.0f, pybind11::arg("histogram_detail") = 0,
        pybind11::arg("max_sigma_bin") = 200, pybind11::arg("threshold"), pybind11::arg("save_stat_cubes") = false,
        pybind11::arg("beam_size_enabled") = false, pybind11::arg("fixed_beam_kernel") = pybind11::none(),
        pybind11::arg("fixed_beam_halfsize") = 0, pybind11::arg("num_threads") = 0,
        pybind11::arg("use_gpu") = false);
}

PYBIND11_MODULE(breifast_helpers, m)
{
#ifdef BREIFAST_HAVE_GPU
  m.attr("has_gpu") = true;
  breifast::gpu::require_cuda_device();
  breifast::gpu::use_pinned_host_memory();
#else
  m.attr("has_gpu") = false;
#endif

  xt::import_numpy();
  init_plane_lot_bindings(m);
}