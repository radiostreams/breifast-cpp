#define XTENSOR_USE_TBB
#define FORCE_IMPORT_ARRAY
#include <optional>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
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

} // namespace

/**
 * @brief pybind11-facing wrapper around process_band_slice_core() (see breifast_core.hpp for the
 * algorithm itself). Converts numpy/pybind11 types at the boundary -- band_slice to a
 * [band_start, band_stop) range, the optional mean/flux/snr-cutoff images from `None`-or-pyarray to
 * std::optional<xt::xarray<float>>, and the ProcessBandSliceResult back into the 13-element tuple
 * this has always returned: `processed` (bool), `snr`, `mask`, `plane_img`, `noisemap`,
 * `excess_noise_factor`, `mednoise`, `noise_globmad`, `minsig`, `maxsig`, and `hist0`/`hist1`/`hist2`
 * (`None` depending on `histogram_detail`).
 *
 * `stokes_cube` is deep-copied into a plain xt::xarray for process_band_slice_core() (which needs a
 * mutable, non-pyarray reference to write SNR back into for single-channel bands) and, if that
 * happened, the mutated channel is copied back into the caller's original array afterwards -- so
 * this still mutates `stokes_cube` in place exactly as before, just via an extra copy at the
 * boundary instead of passing the pyarray straight through.
 */
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
    // process_band_slice_core() only writes the SNR-overwrite back into stokes_cube_local when it
    // actually reaches that point (result.processed == true) -- mirror that here.
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
}
