#!/usr/bin/env python3
"""Scaling benchmark for process_lot on a dumped lot manifest, CPU or GPU.

Usage:
    python3 gpu_bench.py <manifest_dir> --plane-count N --mode cpu|gpu
                         [--threads T] [--repeat R] [--warmup W] [--json PATH]

The manifest directory is the one produced by small_test/dump_lot_for_cpp.py
(manifest.json plus raw .bin arrays), the same input bench.cpp takes.

The full lot is replicated round-robin to N planes so every N has the same
per-plane mix of noise and detection planes. Run one N per process so a CUDA
OOM at a larger N cannot poison the next point.
"""
import argparse
import json
import os
import platform
import socket
import subprocess
import threading
import time

import numpy as np

import breifast_helpers


def load_desc(base_dir, desc):
    if desc is None:
        return None
    path = os.path.join(base_dir, desc["file"])
    return np.fromfile(path, dtype=np.dtype(desc["dtype"])).reshape(desc["shape"])


def build_lot(manifest, base_dir):
    groups = []
    for group in manifest["groups"]:
        groups.append({key: load_desc(base_dir, value) for key, value in group.items()})

    planes = []
    for plane in manifest["planes"]:
        item = {"group_index": int(plane["group_index"]), "stokes_label": str(plane["stokes_label"])}
        for key, value in plane.items():
            if key not in item:
                item[key] = load_desc(base_dir, value)
        planes.append(item)

    kwargs = dict(
        stokes_groups=groups,
        band_slices=[dict(band) for band in manifest["band_slices"]],
        baseline_flux=load_desc(base_dir, manifest["baseline_flux"]),
        subtract_mean=bool(manifest["subtract_mean"]),
        use_rms_noise_estimator=bool(manifest["use_rms_noise_estimator"]),
        boxsize=int(manifest["boxsize"]),
        inner_boxsize=int(manifest["inner_boxsize"]),
        ddenoise_scaling_factor=float(manifest["ddenoise_scaling_factor"]),
        full_bandwidth=float(manifest["full_bandwidth"]),
        reject_noise_threshold=float(manifest["reject_noise_threshold"]),
        histogram_detail=int(manifest["histogram_detail"]),
        max_sigma_bin=int(manifest["max_sigma_bin"]),
        threshold=float(manifest["threshold"]),
        save_stat_cubes=bool(manifest["save_stat_cubes"]),
        beam_size_enabled=bool(manifest["beam_size_enabled"]),
        fixed_beam_kernel=load_desc(base_dir, manifest.get("fixed_beam_kernel")),
        fixed_beam_halfsize=int(manifest["fixed_beam_halfsize"]),
    )
    return planes, kwargs


class GpuMemorySampler:
    def __init__(self):
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._device = visible.split(",")[0] if visible else None

    def _run(self):
        query = ["nvidia-smi"]
        if self._device:
            query += ["-i", self._device]
        query += ["--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(query, text=True)
                self.peak_mb = max(self.peak_mb, int(output.splitlines()[0]))
            except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
                return
            self._stop.wait(0.05)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)


def run_once(planes, kwargs, use_gpu, threads):
    start = time.perf_counter()
    results = breifast_helpers.process_lot(planes=planes, num_threads=threads, use_gpu=use_gpu, **kwargs)
    elapsed = time.perf_counter() - start
    detections = sum(len(result["detections"]["peak_x"]) for result in results)
    return elapsed, detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_dir")
    parser.add_argument("--plane-count", type=int, default=36)
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json")
    parser.add_argument("--expect-detections", type=int)
    args = parser.parse_args()

    use_gpu = args.mode == "gpu"
    if use_gpu and not getattr(breifast_helpers, "has_gpu", False):
        raise SystemExit("GPU mode requested but this build has no GPU support")

    manifest = json.load(open(os.path.join(args.manifest_dir, "manifest.json")))
    source_planes, kwargs = build_lot(manifest, args.manifest_dir)
    planes = [source_planes[index % len(source_planes)] for index in range(args.plane_count)]

    record = {
        "mode": args.mode,
        "host": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "plane_count": args.plane_count,
        "threads": args.threads,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "boxsize": kwargs["boxsize"],
        "times": [],
        "detections": [],
        "peak_gpu_mb": None,
    }

    sampler = GpuMemorySampler() if use_gpu else None
    if sampler is not None:
        sampler.__enter__()
    try:
        for _ in range(args.warmup):
            run_once(planes, kwargs, use_gpu, args.threads)
        for _ in range(args.repeat):
            elapsed, detections = run_once(planes, kwargs, use_gpu, args.threads)
            record["times"].append(elapsed)
            record["detections"].append(detections)
            print(f"{args.mode} planes={args.plane_count} run={len(record['times']) - 1} "
                  f"time={elapsed:.4f}s detections={detections}", flush=True)
    finally:
        if sampler is not None:
            sampler.__exit__()
            record["peak_gpu_mb"] = sampler.peak_mb

    record["median"] = float(np.median(record["times"])) if record["times"] else None
    record["expected_detections"] = args.expect_detections
    if args.expect_detections is None:
        record["success"] = all(count == record["detections"][0] for count in record["detections"])
    else:
        record["success"] = all(count == args.expect_detections for count in record["detections"])
    print(json.dumps(record), flush=True)
    if args.json:
        with open(args.json, "w") as out:
            json.dump(record, out, indent=2)


if __name__ == "__main__":
    main()
