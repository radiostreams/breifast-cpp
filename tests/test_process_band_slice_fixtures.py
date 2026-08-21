"""
Regression test replaying real pipeline data against the current `process_lot`
binding, to confirm the C++ port reproduces breifast.py's own pure-Python results on
a real dataset.

Fixtures are *.npz files under tests/fixtures/ (or $BREIFAST_FIXTURE_DIR), each
holding the exact inputs breifast.py's `verify_cpp` instrumentation passed to a
single real band, plus the pure-python reference outputs it computed for that band
at capture time (`ref_*` keys). These were captured from a real small_test run
(test.ms + deep-image.fits) with BREIFAST_VERIFY_CPP=1, BREIFAST_VERIFY_CPP_LIMIT=0
(the default of 1 only verifies the first band processed) and
BREIFAST_DUMP_FIXTURES set; see BREIFAST_VERIFY_CPP / BREIFAST_VERIFY_CPP_LIMIT /
BREIFAST_DUMP_FIXTURES / BREIFAST_DUMP_FIXTURES_LIMIT at the top of breifast.py.

Only fixtures with at least one real detection (`ref_image_mask.any()`) are
exercised here -- most of a real pipeline's bands are pure noise, and there's
nothing to validate against those beyond what tests/cpp/group3_integration already
covers with synthetic fixtures.

The old capture format predates two `process_lot`/`PlaneBatchGroup`/`PlaneBatchItem`
fields, which aren't reconstructable from what was captured:

- `PlaneBatchGroup.selfnoise_raw`: a separately-computed per-stokes noise map, zero
  -filled here. This means the `self_noise` detection field (which reads directly
  from it) and `beam_chi2`/`boost` (unrelated, always 0 here since beam fitting and
  baseline-flux boosting are both off) cannot be validated by this test -- only the
  fields derivable from what was actually captured (see below) are checked.
- `PlaneBatchItem.nonzero_channels`: approximated as `weights > 0`, which is what it
  is derived from upstream in the real pipeline; safe here since every captured band
  already passed that same nonzero check to be processed (and hence captured) at
  all.

`process_lot` also only returns aggregated per-island detections, not the raw
per-pixel snr/image_mask/plane_img/noisemap maps the retired `process_band_slice`
API returned -- this test instead re-derives the expected per-island values directly
from the real `ref_snr`/`ref_plane_img`/`ref_noisemap`/`ref_excess_noise_factor`
arrays (via the same 4-connectivity labeling, postage-stamp noise estimate, and
`max_to_std_ratio` search that `src/islands.cpp`/`src/plane_lot_dispatch.cpp` use)
and compares those against `process_lot`'s actual detections.
"""

import glob
import math
import os
import re

import numpy as np
import pytest

import breifast_helpers

_FIXTURE_DIR = os.environ.get(
    "BREIFAST_FIXTURE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"),
)
_FIXTURE_PATHS = sorted(glob.glob(os.path.join(_FIXTURE_DIR, "*.npz")))
_FIXTURE_NAME_RE = re.compile(r"\.([A-Za-z]+)\.band(\w+)\.iv\d+\.npz$")


def _none_if_empty(arr):
    return None if arr.size == 0 else arr


def _max_to_std_ratio(n):
    """Python port of PlaneConfig::max_to_std_ratio (plane_lot_dispatch.cpp) -- the
    median of the max of n iid standard-normal samples, found via the same 1000-point
    linear-interpolation search over [-10, 10]."""
    n_samples = 1000
    x_ampl = 10.0
    x_prev = -x_ampl
    f_prev = 0.5 * (1.0 + math.erf(x_prev / math.sqrt(2.0)))
    F_prev = 1.0 - (1.0 - f_prev) ** n
    for i in range(1, n_samples):
        x_val = -x_ampl + i * 2.0 * x_ampl / (n_samples - 1)
        f_val = 0.5 * (1.0 + math.erf(x_val / math.sqrt(2.0)))
        F_val = 1.0 - (1.0 - f_val) ** n
        if F_val >= 0.5:
            return abs(x_prev + (x_val - x_prev) * (0.5 - F_prev) / (F_val - F_prev))
        x_prev, f_prev, F_prev = x_val, f_val, F_val
    return abs(x_prev)


def _label_islands_4conn(mask):
    """Plain-Python 4-connectivity flood fill (no scipy in this environment),
    returns a list of islands, each a list of (y, x) coordinates."""
    points = set(zip(*np.nonzero(mask)))
    visited = set()
    islands = []
    for start in points:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        island = [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (y + dy, x + dx)
                if q in points and q not in visited:
                    visited.add(q)
                    island.append(q)
                    stack.append(q)
        islands.append(island)
    return islands


def _expected_detections(data, boxsize, inner_boxsize):
    mask = data["ref_image_mask"]
    snr = data["ref_snr"]
    plane_img = data["ref_plane_img"]
    noisemap = data["ref_noisemap"]
    excess_noise_factor = data["ref_excess_noise_factor"]
    mednoise = float(data["ref_mednoise"])
    noise_globmad = float(data["ref_noise_globmad"])

    radius = boxsize // 2
    ratio = _max_to_std_ratio((2 * radius + 1) ** 2)

    expected = []
    for island in _label_islands_4conn(mask):
        ys = np.array([p[0] for p in island])
        xs = np.array([p[1] for p in island])
        abs_snr = np.abs(snr[ys, xs])
        peak_i = int(np.argmax(abs_snr))
        y, x = int(ys[peak_i]), int(xs[peak_i])

        assert y - radius >= 0 and x - radius >= 0 and y + radius < mask.shape[0] and x + radius < mask.shape[1], (
            "island too close to the image edge for this test's simplified "
            "(no zero-padding) postage-stamp extraction"
        )
        sub = plane_img[y - radius : y + radius + 1, x - radius : x + radius + 1].copy()
        sub[radius - inner_boxsize : radius + inner_boxsize + 1, radius - inner_boxsize : radius + inner_boxsize + 1] = 0.0
        extreme = max(abs(sub.min()), abs(sub.max()))
        adjacent_noise = extreme / ratio if ratio != 0 else 0.0
        donut = sub[sub != 0.0]
        rms = float(donut.std()) if donut.size else 0.0
        lmad = float(np.median(np.abs(donut)) * 1.4826) if donut.size else 0.0

        flux = float(plane_img[y, x])
        local_noise = float(noisemap[ys, xs].max())

        expected.append(
            dict(
                peak_y=y,
                peak_x=x,
                island_size=len(island),
                flux=flux,
                local_noise=local_noise,
                local_noise_median_ratio=local_noise / mednoise,
                signal_to_noise=float(abs_snr[peak_i]),
                signal_to_noise_rms=(flux / rms) if rms != 0.0 else 0.0,
                signal_to_noise_extended_box=(flux / adjacent_noise) if adjacent_noise != 0.0 else 0.0,
                signal_to_noise_local_mad=(flux / lmad) if lmad != 0.0 else 0.0,
                signal_to_noise_global_mad=flux / noise_globmad,
                excess_noise_factor=float(excess_noise_factor[y, x]),
            )
        )
    return sorted(expected, key=lambda d: (d["peak_y"], d["peak_x"]))


_DETECTION_FIELDS = (
    "peak_y",
    "peak_x",
    "island_size",
    "flux",
    "local_noise",
    "local_noise_median_ratio",
    "signal_to_noise",
    "signal_to_noise_rms",
    "signal_to_noise_extended_box",
    "signal_to_noise_local_mad",
    "signal_to_noise_global_mad",
    "excess_noise_factor",
)


def _actual_detections(detections):
    rows = [
        dict(zip(_DETECTION_FIELDS, values))
        for values in zip(*(detections[field] for field in _DETECTION_FIELDS))
    ]
    return sorted(rows, key=lambda d: (d["peak_y"], d["peak_x"]))


@pytest.mark.skipif(not _FIXTURE_PATHS, reason=f"no fixtures found under {_FIXTURE_DIR}")
@pytest.mark.parametrize("path", _FIXTURE_PATHS, ids=[os.path.basename(p) for p in _FIXTURE_PATHS])
def test_process_lot_matches_real_pipeline_data(path):
    data = np.load(path)

    if not data["ref_image_mask"].any():
        pytest.skip("fixture has no real detections to validate against")

    match = _FIXTURE_NAME_RE.search(os.path.basename(path))
    stokes_label, band_label = match.group(1), match.group(2)

    boxsize = int(data["boxsize"])
    inner_boxsize = 10  # breifast.py's own default; not overridden by this dataset's config.

    band = dict(
        start=int(data["band_start"]),
        stop=int(data["band_stop"]),
        label=band_label,
        selfnoise_scale_factor=float(data["full_bandwidth"])
        / float(data["chanwidth"][int(data["band_start"]) : int(data["band_stop"])].sum()),
    )

    item = dict(
        stokes_cube=data["stokes_cube"],
        weights=data["weights"],
        nonzero_channels=(data["weights"] > 0),
        group_index=0,
        stokes_label=stokes_label,
    )

    group = dict(
        accept_mask=data["accept_mask"],
        selfnoise_offset=data["selfnoise_offset"],
        selfnoise_raw=np.zeros_like(data["selfnoise_offset"]),
        dde_noise=data["dde_noise"],
    )
    mean_image = _none_if_empty(data["mean_image"])
    if mean_image is not None:
        group["mean_image"] = mean_image
    flux_cutoff = _none_if_empty(data["flux_cutoff"])
    if flux_cutoff is not None:
        group["flux_cutoff"] = flux_cutoff
    snr_cutoff = _none_if_empty(data["snr_cutoff"])
    if snr_cutoff is not None:
        group["snr_cutoff"] = snr_cutoff

    [result] = breifast_helpers.process_lot(
        planes=[item],
        stokes_groups=[group],
        band_slices=[band],
        subtract_mean=bool(data["subtract_mean"]),
        use_rms_noise_estimator=bool(data["use_rms"]),
        boxsize=boxsize,
        inner_boxsize=inner_boxsize,
        ddenoise_scaling_factor=float(data["ddenoise_scaling_factor"]),
        full_bandwidth=float(data["full_bandwidth"]),
        reject_noise_threshold=float(data["reject_noise_threshold"]),
        histogram_detail=int(data["histogram_detail"]),
        max_sigma_bin=int(data["max_sigma_bin"]),
        threshold=float(data["threshold"]),
        beam_size_enabled=False,
        num_threads=1,
    )

    np.testing.assert_allclose(result["min_sigma"], data["ref_minsig"], rtol=1e-4, atol=1e-6, err_msg="min_sigma")
    np.testing.assert_allclose(result["max_sigma"], data["ref_maxsig"], rtol=1e-4, atol=1e-6, err_msg="max_sigma")

    expected = _expected_detections(data, boxsize, inner_boxsize)
    got = _actual_detections(result["detections"])

    assert len(got) == len(expected), (
        f"{os.path.basename(path)}: expected {len(expected)} islands "
        f"{[(e['peak_y'], e['peak_x']) for e in expected]}, "
        f"got {len(got)} {[(g['peak_y'], g['peak_x']) for g in got]}"
    )

    tight_fields = (
        "flux",
        "local_noise",
        "local_noise_median_ratio",
        "signal_to_noise",
        "signal_to_noise_global_mad",
        "excess_noise_factor",
    )
    # max_to_std_ratio is computed in float32 by the C++ side and float64 here, so
    # give the fields that depend on it (through adjacent_noise) more slack.
    loose_fields = ("signal_to_noise_rms", "signal_to_noise_local_mad", "signal_to_noise_extended_box")

    for g, e in zip(got, expected):
        assert (g["peak_y"], g["peak_x"]) == (e["peak_y"], e["peak_x"]), f"{os.path.basename(path)}: island position mismatch"
        assert g["island_size"] == e["island_size"], f"{os.path.basename(path)}: island_size mismatch at ({g['peak_y']}, {g['peak_x']})"
        for field in tight_fields:
            np.testing.assert_allclose(g[field], e[field], rtol=1e-4, atol=1e-6, err_msg=f"{os.path.basename(path)}: {field}")
        for field in loose_fields:
            np.testing.assert_allclose(g[field], e[field], rtol=1e-2, atol=1e-6, err_msg=f"{os.path.basename(path)}: {field}")
