#!/usr/bin/env python3
"""Python counterpart of bench: breifast.py's own process_plane() over the bench lot, on a thread pool.

Usage:
    python3 python_bench.py <lot.pkl> [num_threads] [repeat] [replicas]

process_plane() is a closure inside process_residual_cube(), so its source is extracted from breifast.py at runtime
and run unchanged, with the closure variables rebuilt from the lot pickle (BREIFAST_DUMP_LOT output, the same data
bench's manifest was made from). Like bench, the lot's planes are replicated `replicas` times (default 30) and all
items go to the pool at once.
"""
import ast
import inspect
import pickle
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import breifast.breifast as bf
from breifast import LOGGER
from breifast.catalogs import DetectionTable
from breifast.cubes import create_multidimensional_gaussian_kernel, save_to_fits


def load_process_plane(namespace):
    source = inspect.getsource(bf)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "process_plane":
            exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)
            return namespace["process_plane"]
    raise RuntimeError("process_plane not found in breifast.py")


def build_namespace(lot):
    planes = lot["planes"]
    (group,) = lot["stokes_groups"]
    (band,) = lot["band_slices"]
    assert all(p["stokes_label"] == "I" and p["group_index"] == 0 for p in planes)
    assert lot["fixed_beam_kernel"] is None and band["selfnoise_scale_factor"] == 1.0

    def per_interval(key):
        return np.stack([p[key] for p in planes], axis=-1)

    namespace = dict(vars(bf))
    namespace.update(
        LOGGER=LOGGER,
        DetectionTable=DetectionTable,
        create_multidimensional_gaussian_kernel=create_multidimensional_gaussian_kernel,
        save_to_fits=save_to_fits,
        stokes_indices={"I": 0},
        q_stokes_index=None,
        u_stokes_index=None,
        cube=None,
        ds=None,
        spvar_cds=None,
        header=None,
        basename="",
        save_intervals=set(),
        nfreq=planes[0]["stokes_cube"].shape[0],
        nonzero=per_interval("nonzero_channels")[None],
        weights=per_interval("weights")[None],
        bmaj=per_interval("beam_major_axis"),
        bmin=per_interval("beam_minor_axis"),
        bpa=per_interval("beam_position_angle"),
        band_slices=[(slice(band["start"], band["stop"]), band["label"])],
        num_band_slices=1,
        chanwidth=np.array([lot["full_bandwidth"]]),
        full_bandwidth=lot["full_bandwidth"],
        accept_mask=group["accept_mask"][None],
        selfnoise_offset=group["selfnoise_offset"][None],
        selfnoise=group["selfnoise_raw"][None],
        dde_noise=group["dde_noise"][None],
        mean_image=group["mean_image"][None],
        flux_cutoff=group["flux_cutoff"][None],
        snr_cutoff=group["snr_cutoff"][None],
        baseline_flux=lot["baseline_flux"],
        subtract_mean=lot["subtract_mean"],
        noise_estimator="rms" if lot["use_rms_noise_estimator"] else "mad",
        boxsize=lot["boxsize"],
        inner_boxsize=lot["inner_boxsize"],
        ddenoise_scaling_factor=lot["ddenoise_scaling_factor"],
        reject_noise_threshold=lot["reject_noise_threshold"],
        histogram_detail=lot["histogram_detail"],
        max_sigma_bin=lot["max_sigma_bin"],
        histogram_bins=np.arange(-lot["max_sigma_bin"], lot["max_sigma_bin"] + 1),
        threshold=lot["threshold"],
        save_stat_cubes=lot["save_stat_cubes"],
        beam_size=lot["beam_size_enabled"],
        beam_kernel=None,
        beam_halfsize=0,
        _python_plane_compute_times=[],
    )
    return namespace


def main():
    pkl_path = sys.argv[1]
    num_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 72
    repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    replicas = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    with open(pkl_path, "rb") as f:
        lot = pickle.load(f)
    namespace = build_namespace(lot)
    process_plane = load_process_plane(namespace)
    n_planes = len(lot["planes"])
    print(f"loaded {n_planes} planes x {replicas} replicas from {pkl_path}, num_threads={num_threads}, "
          f"repeat={repeat}", flush=True)

    times = []
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        for i in range(repeat):
            # process_plane overwrites single-channel bands with SNR, so every item gets a fresh copy
            cubes = [lot["planes"][j]["stokes_cube"][None].copy() for _ in range(replicas) for j in range(n_planes)]
            t0 = time.perf_counter()
            results = list(pool.map(lambda j: process_plane(j % n_planes, "I", cubes[j]), range(len(cubes))))
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            errors = sum(isinstance(r[2], Exception) for r in results)
            n_detections = sum(len(r[2]) for r in results if r[2] is not None and not isinstance(r[2], Exception))
            print(f"  iteration {i}: {len(results)} items, {n_detections} detections, {errors} errors, "
                  f"compute time = {elapsed:.3f}s", flush=True)
            del cubes

    print(f"python process_plane: {repeat} iterations, total = {sum(times):.3f}s, mean = {sum(times) / repeat:.3f}s")


if __name__ == "__main__":
    main()
