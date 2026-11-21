"""process_plane_spectron gives identical results whatever the memory order of its input planes.

The current forward-difference and convolution stores keep (l, m) last; procres reads them by name in (m, l)
order, so the kernel receives transposed views: same values, other strides. The kernel must accept them and
produce exactly what it produces from C-contiguous planes.
"""
import numpy as np
import pytest

breifast_helpers = pytest.importorskip("breifast_helpers")

NY, NX, NFREQ = 96, 80, 2          # deliberately not square: a transpose bug cannot hide
SLICES = [(0, 1, "B0"), (1, 2, "B1"), (0, 2, "B0~1")]


def transposed_memory(a):
    """Same values and axes as `a`, with its last two axes stored transposed (as an (l, m) store read as (m, l))."""
    return np.swapaxes(np.ascontiguousarray(np.swapaxes(a, -1, -2)), -1, -2)


def make_config(estimator):
    cfg = breifast_helpers.SpectronConfig()
    cfg.ny, cfg.nx, cfg.nfreq = NY, NX, NFREQ
    cfg.set_band_slices(SLICES)
    cfg.subtract_mean = True
    cfg.noise_estimator = estimator
    cfg.boxsize = 20
    cfg.inner_boxsize = 3
    cfg.full_bandwidth = 2.0
    cfg.reject_noise_threshold = 3.0
    cfg.histogram_detail = 2
    cfg.nominal_peaks = True
    cfg.threshold = 5.0
    cfg.flood_cell_max = 8
    cfg.flood_cell_size = 32
    cfg.edge_margin = 4
    cfg.min_sigma_rms = 3.0
    cfg.beam_size = True
    cfg.dip_x, cfg.dip_y, cfg.dip_flux = [30.0], [40.0], [0.01]
    cfg.dipole_radius = 2.5
    return cfg


def make_inputs(seed=1):
    rng = np.random.default_rng(seed)
    cube = rng.normal(0, 1e-3, (1, NFREQ, NY, NX)).astype(np.float32)
    yy, xx = np.mgrid[:NY, :NX]
    for (y0, x0, amp) in ((30, 50, 0.02), (60, 20, -0.015), (45, 35, 0.012)):   # sources, one negative
        cube[0, :] += (amp * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 1.5 ** 2))).astype(np.float32)
    images = dict(
        accept_mask=np.ones((NFREQ, NY, NX), bool),
        mean_image=rng.normal(0, 1e-5, (NFREQ, NY, NX)).astype(np.float32),
        snr_cutoff=np.zeros((NFREQ, NY, NX), np.float32),
        flux_cutoff=np.full((NFREQ, NY, NX), 1e-6, np.float64),
        selfnoise_offset=np.abs(rng.normal(0, 1e-8, (NY, NX))).astype(np.float32),
        dde_noise=np.abs(rng.normal(0, 1e-8, (NY, NX))).astype(np.float32),
        nm_noise=np.zeros((NY, NX), np.float32),
        noise_scale_sqrt=(1 + 0.05 * rng.random((NY, NX))).astype(np.float64),
        selfnoise=np.abs(rng.normal(0, 1e-3, (NY, NX))).astype(np.float32),
        baseline_flux=np.full((NY, NX), 1e-3, np.float32),
    )
    images["accept_mask"][:, :, :3] = False
    return cube, images


def run(cfg, cube, images):
    return breifast_helpers.process_plane_spectron(
        cfg, cube, 0, None, np.ones(NFREQ, bool), np.ones(NFREQ, np.float32), np.ones(NFREQ, np.float32),
        images["accept_mask"], images["mean_image"], images["snr_cutoff"], images["flux_cutoff"],
        images["selfnoise_offset"], images["dde_noise"], images["nm_noise"], None, images["noise_scale_sqrt"],
        images["selfnoise"], images["baseline_flux"],
        np.full(NFREQ, 1.2, np.float32), np.full(NFREQ, 1.0, np.float32), np.zeros(NFREQ, np.float32),
        [5.0] * len(SLICES), None, None)


def assert_same(a, b, what=""):
    if isinstance(a, dict):
        assert set(a) == set(b), what
        for k in a:
            assert_same(a[k], b[k], f"{what}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), what
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{what}[{i}]")
    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape, what
        if a.dtype.kind in "fc":
            np.testing.assert_array_equal(a, b, err_msg=what)   # NaNs compare equal here
        else:
            assert (a == b).all(), what
    else:
        assert a == b or (a != a and b != b), f"{what}: {a!r} vs {b!r}"


@pytest.mark.parametrize("estimator", ["mad", "rms"])
def test_transposed_planes_give_identical_results(estimator):
    cube, images = make_inputs()
    ref_cube = cube.copy()
    ref = run(make_config(estimator), ref_cube, images)
    assert len(ref["detections"]["x"]) > 0, "the test field must produce detections"

    cfg = make_config(estimator)                      # one run: its cache is used by both calls below
    t_images = {k: transposed_memory(v) for k, v in images.items()}
    for call in range(2):                             # the second call takes the per-run images from the cache
        t_cube = transposed_memory(cube)
        assert not t_cube[0, 0].flags.c_contiguous
        res = run(cfg, t_cube, t_images)
        assert_same(ref, res, f"call {call}")
        # the cube is changed in place as with C-contiguous planes (single-band slices replaced by SNR)
        np.testing.assert_array_equal(np.asarray(t_cube), ref_cube)


def test_outputs_must_be_contiguous():
    """Arrays the kernel writes into (the accumulators) are refused in any other memory order."""
    cube, images = make_inputs()
    cfg = make_config("mad")
    pb = transposed_memory(np.zeros((NFREQ, NY, NX)))
    with pytest.raises(ValueError, match="contiguous"):
        breifast_helpers.process_plane_spectron(
            cfg, cube, 0, None, np.ones(NFREQ, bool), np.ones(NFREQ, np.float32), np.ones(NFREQ, np.float32),
            images["accept_mask"], images["mean_image"], images["snr_cutoff"], images["flux_cutoff"],
            images["selfnoise_offset"], images["dde_noise"], images["nm_noise"], None, images["noise_scale_sqrt"],
            images["selfnoise"], images["baseline_flux"],
            np.full(NFREQ, 1.2, np.float32), np.full(NFREQ, 1.0, np.float32), np.zeros(NFREQ, np.float32),
            [5.0] * len(SLICES), pb, None)
