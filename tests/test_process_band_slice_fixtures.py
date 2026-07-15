"""
Regression tests replaying real pipeline data captured by breifast.py's
verify_cpp instrumentation (see BREIFAST_DUMP_FIXTURES in breifast.py)
against breifast_helpers.process_band_slice directly, without needing the
full python pipeline or scipy available.

Fixtures are *.npz files under tests/fixtures/ (or $BREIFAST_FIXTURE_DIR),
each holding the exact inputs passed to process_band_slice for one real
band, plus the pure-python reference outputs computed at capture time. To
add more: run the real pipeline with BREIFAST_VERIFY_CPP=1 (the default) and
BREIFAST_DUMP_FIXTURES=some/dir set, then copy the resulting *.npz files
into tests/fixtures/.

Environment variables read by breifast.py that control this (documented
here since this test suite is what actually consumes their output):

- BREIFAST_VERIFY_CPP: master on/off switch for the verify_cpp machinery
  (the C++ call, the python-vs-C++ comparison/logging, and the fixture
  capture below). Defaults to "1" (on). Set to "0" to skip it entirely,
  e.g. on a production run once the port is trusted -- running both
  implementations roughly doubles per-band wall-clock time. Fixture
  capture requires this to be on, since fixtures are only ever written
  from inside the comparison code path.
- BREIFAST_DUMP_FIXTURES: directory to write captured *.npz fixtures into.
  Unset (the default) disables capture even if BREIFAST_VERIFY_CPP=1.
- BREIFAST_DUMP_FIXTURES_LIMIT: max number of fixtures to capture per
  breifast.py process (default 5), so a full pipeline run doesn't fill the
  fixtures directory.
"""
import glob
import os

import numpy as np
import pytest

import breifast_helpers

_FIXTURE_DIR = os.environ.get(
    "BREIFAST_FIXTURE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"),
)
_FIXTURE_PATHS = sorted(glob.glob(os.path.join(_FIXTURE_DIR, "*.npz")))


def _none_if_empty(arr):
    return None if arr.size == 0 else arr


@pytest.mark.skipif(not _FIXTURE_PATHS, reason=f"no fixtures found under {_FIXTURE_DIR}")
@pytest.mark.parametrize("path", _FIXTURE_PATHS, ids=[os.path.basename(p) for p in _FIXTURE_PATHS])
def test_matches_captured_python_reference(path):
    data = np.load(path)

    band_slice = slice(int(data["band_start"]), int(data["band_stop"]))
    mean_image = _none_if_empty(data["mean_image"])
    flux_cutoff = _none_if_empty(data["flux_cutoff"])
    snr_cutoff = _none_if_empty(data["snr_cutoff"])

    (processed, snr, image_mask, plane_img, noisemap, excess_noise_factor,
     mednoise, noise_globmad, minsig, maxsig, hist0, hist1, hist2) = breifast_helpers.process_band_slice(
        data["stokes_cube"].copy(),
        data["weights"],
        data["accept_mask"],
        data["chanwidth"],
        band_slice,
        bool(data["is_single_chan"]),
        bool(data["subtract_mean"]),
        mean_image,
        bool(data["use_rms"]),
        int(data["boxsize"]),
        data["selfnoise_offset"],
        data["dde_noise"],
        float(data["ddenoise_scaling_factor"]),
        float(data["full_bandwidth"]),
        flux_cutoff,
        snr_cutoff,
        float(data["reject_noise_threshold"]),
        int(data["histogram_detail"]),
        int(data["max_sigma_bin"]),
        float(data["threshold"]),
    )

    assert processed, "cpp skipped a band the original python pipeline processed"

    for name, cpp_val, ref_key in [
        ("snr", snr, "ref_snr"),
        ("image_mask", image_mask, "ref_image_mask"),
        ("plane_img", plane_img, "ref_plane_img"),
        ("noisemap", noisemap, "ref_noisemap"),
        ("excess_noise_factor", excess_noise_factor, "ref_excess_noise_factor"),
        ("mednoise", mednoise, "ref_mednoise"),
        ("noise_globmad", noise_globmad, "ref_noise_globmad"),
        ("minsig", minsig, "ref_minsig"),
        ("maxsig", maxsig, "ref_maxsig"),
    ]:
        cpp_arr = np.asarray(cpp_val)
        ref_arr = np.asarray(data[ref_key])
        if cpp_arr.dtype == bool:
            assert np.array_equal(cpp_arr, ref_arr), f"'{name}' mismatch in {os.path.basename(path)}"
        else:
            np.testing.assert_allclose(
                cpp_arr.astype(float), ref_arr.astype(float),
                rtol=1e-5, atol=1e-6, equal_nan=True,
                err_msg=f"'{name}' mismatch in {os.path.basename(path)}",
            )
