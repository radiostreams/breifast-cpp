#define XTENSOR_USE_TBB
#define FORCE_IMPORT_ARRAY
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pyarray.hpp>

#include "breifast_core.hpp"

namespace
{

std::optional<xt::xarray<float>> to_optional_xarray(const pybind11::object &obj)
{
  if (obj.is_none())
  {
    return std::nullopt;
  }
  return xt::xarray<float>(obj.cast<xt::pyarray<float>>());
}

std::optional<xt::xarray<double>> to_optional_xarray_double(const pybind11::object &obj)
{
  if (obj.is_none())
  {
    return std::nullopt;
  }
  return xt::xarray<double>(obj.cast<xt::pyarray<double>>());
}

pybind11::dict build_process_plane_result_dict(const ProcessPlaneResult &result, int histogram_detail)
{
  pybind11::dict out;
  out["any_data"] = result.any_data;
  out["x"] = pybind11::cast(result.detections.x);
  out["y"] = pybind11::cast(result.detections.y);
  out["band"] = pybind11::cast(result.detections.band_label);
  out["peak_flux"] = pybind11::cast(result.detections.peak_flux);
  out["sigma"] = pybind11::cast(result.detections.sigma);
  out["sigma_ext"] = pybind11::cast(result.detections.sigma_ext);
  out["sigma_rms"] = pybind11::cast(result.detections.sigma_rms);
  out["sigma_lmad"] = pybind11::cast(result.detections.sigma_lmad);
  out["sigma_gmad"] = pybind11::cast(result.detections.sigma_gmad);
  out["boost"] = pybind11::cast(result.detections.boost);
  out["size"] = pybind11::cast(result.detections.size);
  out["ln"] = pybind11::cast(result.detections.local_noise);
  out["ln_x_curmed"] = pybind11::cast(result.detections.ln_x_curmed);
  out["selfnoise"] = pybind11::cast(result.detections.selfnoise_value);
  out["enf"] = pybind11::cast(result.detections.excess_noise_factor_value);
  out["beamchi2"] = pybind11::cast(result.detections.beamchi2);
  out["minsig"] = result.minsig.has_value() ? pybind11::cast(*result.minsig) : pybind11::none();
  out["maxsig"] = result.maxsig.has_value() ? pybind11::cast(*result.maxsig) : pybind11::none();
  out["hist0"] = histogram_detail > 1 ? pybind11::cast(result.hist0) : pybind11::none();
  out["hist1"] = histogram_detail > 1 ? pybind11::cast(result.hist1) : pybind11::none();
  out["hist2"] = histogram_detail > 0 ? pybind11::cast(result.hist2) : pybind11::none();
  out["mednoises"] = pybind11::cast(result.mednoises);

  if (result.variance_stats.has_value())
  {
    pybind11::dict variance_stats;
    variance_stats["spvar"] = xt::pyarray<double>(result.variance_stats->spvar);
    variance_stats["spskew"] = xt::pyarray<double>(result.variance_stats->spskew);
    variance_stats["spkurt"] = xt::pyarray<double>(result.variance_stats->spkurt);
    variance_stats["counts"] = result.variance_stats->counts;
    if (result.variance_stats->timevar.has_value())
    {
      variance_stats["timevar"] = xt::pyarray<float>(*result.variance_stats->timevar);
    }
    else
    {
      variance_stats["timevar"] = pybind11::none();
    }
    out["variance_stats"] = variance_stats;
  }
  else
  {
    out["variance_stats"] = pybind11::none();
  }

  return out;
}

xt::xarray<float> require_xarray(const pybind11::dict &d, const char *key)
{
  return xt::xarray<float>(d[key].cast<xt::pyarray<float>>());
}

xt::xarray<bool> require_xarray_bool(const pybind11::dict &d, const char *key)
{
  return xt::xarray<bool>(d[key].cast<xt::pyarray<bool>>());
}

std::optional<xt::xarray<float>> optional_xarray_key(const pybind11::dict &d, const char *key)
{
  if (!d.contains(key))
  {
    return std::nullopt;
  }
  return to_optional_xarray(d[key]);
}

} // namespace

pybind11::tuple
process_band_slice(xt::pyarray<float> &stokes_cube, xt::pyarray<float> &weights, xt::pyarray<bool> &accept_mask,
                   xt::pyarray<float> &channel_width, const pybind11::slice &band_slice, bool is_single_channel,
                   bool subtract_mean, pybind11::object mean_image_obj, bool use_rms_noise_estimator, int boxsize,
                   xt::pyarray<float> &selfnoise_offset, xt::pyarray<float> &dde_noise, float ddenoise_scaling_factor,
                   float full_bandwidth, pybind11::object flux_cutoff_obj, pybind11::object snr_cutoff_obj,
                   float reject_noise_threshold, int histogram_detail, int max_sigma_bin, float threshold)
{
  const auto n_frequencies = stokes_cube.shape(0);

  Py_ssize_t band_start, band_stop, band_step, band_length;
  band_slice.compute(n_frequencies, &band_start, &band_stop, &band_step, &band_length);
  if (band_step != 1)
  {
    throw std::invalid_argument("process_band_slice: only unit-step band slices are supported");
  }

  xt::xarray<float> stokes_cube_local = stokes_cube;

  ProcessBandSliceResult result = process_band_slice_core(
      stokes_cube_local, xt::xarray<float>(weights), xt::xarray<bool>(accept_mask), xt::xarray<float>(channel_width),
      band_start, band_stop, is_single_channel, subtract_mean, to_optional_xarray(mean_image_obj),
      use_rms_noise_estimator, boxsize, xt::xarray<float>(selfnoise_offset), xt::xarray<float>(dde_noise),
      ddenoise_scaling_factor, full_bandwidth, to_optional_xarray(flux_cutoff_obj), to_optional_xarray(snr_cutoff_obj),
      reject_noise_threshold, histogram_detail, max_sigma_bin, threshold);

  if (is_single_channel)
  {
    if (result.processed)
    {
      xt::view(stokes_cube, band_start, xt::all(), xt::all()) = xt::view(stokes_cube_local, band_start, xt::all(), xt::all());
    }
  }

  if (!result.processed)
  {
    return pybind11::make_tuple(false, pybind11::none(), pybind11::none(), pybind11::none(), pybind11::none(),
                                pybind11::none(), result.mednoise, result.noise_globmad, pybind11::none(),
                                pybind11::none(), pybind11::none(), pybind11::none(), pybind11::none());
  }

  return pybind11::make_tuple(true, xt::pyarray<float>(result.snr), xt::pyarray<bool>(result.mask),
                              xt::pyarray<float>(result.plane_img), xt::pyarray<float>(result.noisemap),
                              xt::pyarray<float>(result.excess_noise_factor), result.mednoise, result.noise_globmad,
                              result.minsig, result.maxsig,
                              histogram_detail > 1 ? pybind11::cast(result.hist0) : pybind11::none(),
                              histogram_detail > 1 ? pybind11::cast(result.hist1) : pybind11::none(),
                              histogram_detail > 0 ? pybind11::cast(result.hist2) : pybind11::none());
}

pybind11::list process_plane_batch(pybind11::list stokes_groups, xt::pyarray<float> &channel_width,
                                   pybind11::object baseline_flux_obj,
                                   std::vector<std::pair<pybind11::slice, std::string>> band_slices,
                                   bool subtract_mean, bool use_rms_noise_estimator, int boxsize, int inner_boxsize,
                                   float ddenoise_scaling_factor, float full_bandwidth, float reject_noise_threshold,
                                   int histogram_detail, int max_sigma_bin, float threshold, bool save_stat_cubes,
                                   bool beam_size_enabled, pybind11::object fixed_beam_kernel_obj,
                                   long fixed_beam_halfsize, int num_threads)
{
  std::vector<PlaneBatchGroup> groups;
  groups.reserve(stokes_groups.size());
  std::vector<PlaneBatchItem> items;
  std::vector<xt::pyarray<float>> item_pyarrays;

  for (auto group_handle : stokes_groups)
  {
    pybind11::dict group_dict = group_handle.cast<pybind11::dict>();
    PlaneBatchGroup group;
    group.accept_mask = require_xarray_bool(group_dict, "accept_mask");
    group.mean_image = optional_xarray_key(group_dict, "mean_image");
    group.selfnoise_offset = require_xarray(group_dict, "selfnoise_offset");
    group.selfnoise_raw = require_xarray(group_dict, "selfnoise_raw");
    group.dde_noise = require_xarray(group_dict, "dde_noise");
    group.flux_cutoff = optional_xarray_key(group_dict, "flux_cutoff");
    group.snr_cutoff = optional_xarray_key(group_dict, "snr_cutoff");
    const size_t group_index = groups.size();
    groups.push_back(std::move(group));

    pybind11::list item_list = group_dict["items"].cast<pybind11::list>();
    for (auto item_handle : item_list)
    {
      pybind11::dict item_dict = item_handle.cast<pybind11::dict>();
      xt::pyarray<float> stokes_cube_pyarray = item_dict["stokes_cube"].cast<xt::pyarray<float>>();

      PlaneBatchItem item;
      item.group_index = group_index;
      item.stokes_cube = xt::xarray<float>(stokes_cube_pyarray);
      item.weights = require_xarray(item_dict, "weights");
      item.nonzero_channels = require_xarray_bool(item_dict, "nonzero_channels");
      item.bmaj = optional_xarray_key(item_dict, "bmaj");
      item.bmin = optional_xarray_key(item_dict, "bmin");
      item.bpa = optional_xarray_key(item_dict, "bpa");

      item_pyarrays.push_back(std::move(stokes_cube_pyarray));
      items.push_back(std::move(item));
    }
  }

  const xt::xarray<float> channel_width_local(channel_width);
  const std::optional<xt::xarray<float>> baseline_flux = to_optional_xarray(baseline_flux_obj);

  const Py_ssize_t n_frequencies = items.empty() ? 0 : static_cast<Py_ssize_t>(items[0].stokes_cube.shape(0));
  std::vector<BandSlice> resolved_band_slices;
  resolved_band_slices.reserve(band_slices.size());
  for (auto &[py_slice, label] : band_slices)
  {
    Py_ssize_t start, stop, step, length;
    py_slice.compute(n_frequencies, &start, &stop, &step, &length);
    if (step != 1)
    {
      throw std::invalid_argument("process_plane_batch: only unit-step band slices are supported");
    }
    resolved_band_slices.push_back(BandSlice{static_cast<long>(start), static_cast<long>(stop), label});
  }

  PlaneConfig config;
  config.band_slices = std::move(resolved_band_slices);
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
  config.fixed_beam_kernel = to_optional_xarray_double(fixed_beam_kernel_obj);
  config.fixed_beam_halfsize = fixed_beam_halfsize;

  std::vector<PlaneBatchItemResult> results;
  {
    pybind11::gil_scoped_release release;
    results = process_plane_batch_core(items, groups, channel_width_local, baseline_flux, config, num_threads);
  }

  pybind11::object runtime_error_cls = pybind11::module_::import("builtins").attr("RuntimeError");
  pybind11::list out_list;
  for (size_t i = 0; i < items.size(); ++i)
  {
    for (long f = 0; f < static_cast<long>(item_pyarrays[i].shape(0)); ++f)
    {
      xt::view(item_pyarrays[i], f, xt::all(), xt::all()) = xt::view(items[i].stokes_cube, f, xt::all(), xt::all());
    }

    if (results[i].has_error)
    {
      out_list.append(runtime_error_cls(results[i].error_message));
      continue;
    }
    out_list.append(build_process_plane_result_dict(results[i].result, histogram_detail));
  }

  return out_list;
}

PYBIND11_MODULE(breifast_helpers, m)
{
  xt::import_numpy();
  m.def("process_band_slice", &process_band_slice, pybind11::arg("stokes_cube"), pybind11::arg("weights"),
        pybind11::arg("accept_mask"), pybind11::arg("channel_width"), pybind11::arg("band_slice"),
        pybind11::arg("is_single_channel"), pybind11::arg("subtract_mean"),
        pybind11::arg("mean_image") = pybind11::none(), pybind11::arg("use_rms_noise_estimator") = true,
        pybind11::arg("boxsize"), pybind11::arg("selfnoise_offset"), pybind11::arg("dde_noise"),
        pybind11::arg("ddenoise_scaling_factor") = 1.0f, pybind11::arg("full_bandwidth"),
        pybind11::arg("flux_cutoff") = pybind11::none(), pybind11::arg("snr_cutoff") = pybind11::none(),
        pybind11::arg("reject_noise_threshold") = 0.0f, pybind11::arg("histogram_detail") = 0,
        pybind11::arg("max_sigma_bin") = 200, pybind11::arg("threshold"));

  m.def("process_plane_batch", &process_plane_batch, pybind11::arg("stokes_groups"),
        pybind11::arg("channel_width"), pybind11::arg("baseline_flux") = pybind11::none(),
        pybind11::arg("band_slices"), pybind11::arg("subtract_mean"),
        pybind11::arg("use_rms_noise_estimator") = true, pybind11::arg("boxsize"),
        pybind11::arg("inner_boxsize") = 10, pybind11::arg("ddenoise_scaling_factor") = 1.0f,
        pybind11::arg("full_bandwidth"), pybind11::arg("reject_noise_threshold") = 0.0f,
        pybind11::arg("histogram_detail") = 0, pybind11::arg("max_sigma_bin") = 200, pybind11::arg("threshold"),
        pybind11::arg("save_stat_cubes") = false, pybind11::arg("beam_size_enabled") = false,
        pybind11::arg("fixed_beam_kernel") = pybind11::none(), pybind11::arg("fixed_beam_halfsize") = 0,
        pybind11::arg("num_threads") = 0);
}
