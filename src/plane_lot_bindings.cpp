#include "plane_lot_bindings.hpp"
#include "plane_lot_dispatch.hpp"

#include <pybind11/stl.h>
#include <xtensor-python/pyarray.hpp>

// Pybind11 bindings for run_lot_core() (plane_lot_dispatch.hpp) -- deliberately kept separate from
// breifast_helpers.cpp's marshaling so this file only has to answer "how does one Python lot turn
// into a LotBatch", not the rest of the module's bindings.
//

namespace
{

template <typename T> xt::xarray<T> require_xarray(const pybind11::dict &d, const char *key)
{
  return xt::xarray<T>(d[key].cast<xt::pyarray<T>>());
}

template <typename T> std::optional<xt::xarray<T>> optional_xarray_key(const pybind11::dict &d, const char *key)
{
  if (!d.contains(key))
  {
    return std::nullopt;
  }
  return xt::xarray<T>(d[key].cast<xt::pyarray<T>>());
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

PlaneBatchItem to_plane_batch_item(const pybind11::dict &plane_dict)
{
  PlaneBatchItem item;
  item.stokes_cube = require_xarray<float>(plane_dict, "stokes_cube");
  item.weights = require_xarray<float>(plane_dict, "weights");
  item.nonzero_channels = require_xarray<bool>(plane_dict, "nonzero_channels");
  item.group_index = plane_dict["group_index"].cast<size_t>();
  item.stokes_label = plane_dict["stokes_label"].cast<std::string>();
  item.beam_major_axis = optional_xarray_key<float>(plane_dict, "beam_major_axis");
  item.beam_minor_axis = optional_xarray_key<float>(plane_dict, "beam_minor_axis");
  item.beam_position_angle = optional_xarray_key<float>(plane_dict, "beam_position_angle");
  return item;
}

PlaneBatchGroup to_plane_batch_group(const pybind11::dict &group_dict)
{
  PlaneBatchGroup group;
  group.accept_mask = require_xarray<bool>(group_dict, "accept_mask");
  group.mean_image = optional_xarray_key<float>(group_dict, "mean_image");
  group.selfnoise_offset = require_xarray<float>(group_dict, "selfnoise_offset");
  group.selfnoise_raw = require_xarray<float>(group_dict, "selfnoise_raw");
  group.dde_noise = require_xarray<float>(group_dict, "dde_noise");
  group.flux_cutoff = optional_xarray_key<float>(group_dict, "flux_cutoff");
  group.snr_cutoff = optional_xarray_key<float>(group_dict, "snr_cutoff");
  return group;
}

std::optional<xt::xarray<float>> to_optional_xarray(const pybind11::object &obj)
{
  if (obj.is_none())
  {
    return std::nullopt;
  }
  return xt::xarray<float>(obj.cast<xt::pyarray<float>>());
}

pybind11::object to_python_result(const ProcessPlaneResult &result)
{
  pybind11::dict out;
  out["any_data"] = result.any_data;
  out["min_sigma"] = result.min_sigma;
  out["max_sigma"] = result.max_sigma;
  out["histogram0"] = result.histogram0;
  out["histogram1"] = result.histogram1;
  out["histogram2"] = result.histogram2;
  out["median_noises"] = result.median_noises;

  pybind11::dict detections;
  detections["peak_x"] = result.detections.peak_x;
  detections["peak_y"] = result.detections.peak_y;
  detections["island_size"] = result.detections.island_size;
  detections["stokes_label"] = result.detections.stokes_label;
  detections["band_label"] = result.detections.band_label;
  detections["local_noise"] = result.detections.local_noise;
  detections["local_noise_median_ratio"] = result.detections.local_noise_median_ratio;
  detections["signal_to_noise"] = result.detections.signal_to_noise;
  detections["signal_to_noise_rms"] = result.detections.signal_to_noise_rms;
  detections["signal_to_noise_extended_box"] = result.detections.signal_to_noise_extended_box;
  detections["signal_to_noise_local_mad"] = result.detections.signal_to_noise_local_mad;
  detections["signal_to_noise_global_mad"] = result.detections.signal_to_noise_global_mad;
  detections["self_noise"] = result.detections.self_noise;
  detections["excess_noise_factor"] = result.detections.excess_noise_factor;
  detections["beam_chi2"] = result.detections.beam_chi2;
  detections["boost"] = result.detections.boost;
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
                           bool save_stat_cubes, bool beam_size_enabled, int num_threads)
{
  LotBatch lot;
  for (auto plane_handle : planes)
  {
    lot.planes.push_back(to_plane_batch_item(plane_handle.cast<pybind11::dict>()));
  }
  for (auto group_handle : stokes_groups)
  {
    lot.groups.push_back(to_plane_batch_group(group_handle.cast<pybind11::dict>()));
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
  config.baseline_flux = to_optional_xarray(baseline_flux_obj);
  for (auto band_slice_handle : band_slices)
  {
    config.band_slices.push_back(to_band_slice(band_slice_handle.cast<pybind11::dict>()));
  }

  std::vector<ProcessPlaneResult> results;
  {
    pybind11::gil_scoped_release release;
    results = run_lot_core(lot, config, num_threads);
  }

  pybind11::list out;
  for (const ProcessPlaneResult &result : results)
  {
    out.append(to_python_result(result));
  }
  return out;
}

} // namespace

void init_plane_lot_bindings(pybind11::module_ &m)
{
  m.def("process_lot", &process_lot, pybind11::arg("planes"), pybind11::arg("stokes_groups"),
        pybind11::arg("band_slices"), pybind11::arg("baseline_flux") = pybind11::none(), pybind11::arg("subtract_mean"),
        pybind11::arg("use_rms_noise_estimator") = true, pybind11::arg("boxsize"), pybind11::arg("inner_boxsize") = 10,
        pybind11::arg("ddenoise_scaling_factor") = 1.0f, pybind11::arg("full_bandwidth"),
        pybind11::arg("reject_noise_threshold") = 0.0f, pybind11::arg("histogram_detail") = 0,
        pybind11::arg("max_sigma_bin") = 200, pybind11::arg("threshold"), pybind11::arg("save_stat_cubes") = false,
        pybind11::arg("beam_size_enabled") = false, pybind11::arg("num_threads") = 0);
}
