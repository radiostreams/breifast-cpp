#!/usr/bin/env python3
"""Build a multi-channel bench lot from lot_diff36_cpp.

Usage: make_synthetic_lot.py <src_dir> <dst_dir> [nfreq=4] [n_planes=8]

Channel k of each plane is the source image plus 10% Gaussian noise. Group arrays (mean, cutoffs, accept mask) repeat
the source channel. The slice list follows breifast.py (single channels, then 2, 4, ... channel sub-bands) with equal
channel widths.
"""
import copy
import json
import os
import sys

import numpy as np

NOISE_FRACTION = 0.1


def band_slices(nfreq, full_bandwidth):
    slices = [(i, i + 1) for i in range(nfreq)]
    stepping = 2
    while stepping <= nfreq:
        for b0 in range(0, nfreq - stepping + 1, stepping // 2):
            b1 = min(b0 + stepping, nfreq)
            slices.append((b0, b1))
            if b1 == nfreq:
                break
        stepping *= 2
    return [
        dict(start=b0, stop=b1, selfnoise_scale_factor=float(nfreq / (b1 - b0)),
             label=f"B{b0}" if b1 - b0 == 1 else f"B{b0}~{b1 - 1}")
        for b0, b1 in slices
    ]


def read(src, entry):
    return np.fromfile(os.path.join(src, entry["file"]), dtype=entry["dtype"]).reshape(entry["shape"])


def write(dst, name, array, dtype):
    array = np.ascontiguousarray(array, dtype=dtype)
    array.tofile(os.path.join(dst, name))
    return dict(file=name, shape=list(array.shape), dtype=dtype)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    nfreq = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_planes = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(dst, exist_ok=True)
    rng = np.random.default_rng(0)

    manifest = json.load(open(os.path.join(src, "manifest.json")))
    out = copy.deepcopy(manifest)
    out["band_slices"] = band_slices(nfreq, manifest["full_bandwidth"])

    out["baseline_flux"] = write(dst, "baseline_flux.bin", read(src, manifest["baseline_flux"]), "float32")

    group = manifest["groups"][0]
    new_group = {}
    for key in ("selfnoise_offset", "selfnoise_raw", "dde_noise"):
        new_group[key] = write(dst, f"group_0_{key}.bin", read(src, group[key]), "float32")
    for key, dtype in (("accept_mask", "bool"), ("mean_image", "float32"), ("flux_cutoff", "float32"),
                       ("snr_cutoff", "float32")):
        base = read(src, group[key])
        new_group[key] = write(dst, f"group_0_{key}.bin", np.repeat(base, nfreq, axis=0), dtype)
    out["groups"] = [new_group]

    out["planes"] = []
    for i, plane in enumerate(manifest["planes"][:n_planes]):
        base = read(src, plane["stokes_cube"])[0]
        sigma = NOISE_FRACTION * float(np.std(base))
        cube = base[None] + sigma * rng.standard_normal((nfreq,) + base.shape, dtype=np.float32)
        entry = dict(group_index=0, stokes_label=plane["stokes_label"])
        entry["stokes_cube"] = write(dst, f"plane_{i}_stokes_cube.bin", cube, "float32")
        weight = float(read(src, plane["weights"])[0])
        entry["weights"] = write(dst, f"plane_{i}_weights.bin", np.full(nfreq, weight), "float32")
        entry["nonzero_channels"] = write(dst, f"plane_{i}_nonzero_channels.bin", np.ones(nfreq), "bool")
        for key in ("beam_major_axis", "beam_minor_axis", "beam_position_angle"):
            value = float(read(src, plane[key])[0])
            entry[key] = write(dst, f"plane_{i}_{key}.bin", np.full(nfreq, value), "float32")
        out["planes"].append(entry)

    json.dump(out, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
    print(f"{len(out['planes'])} planes, {nfreq} channels, {len(out['band_slices'])} slices")


if __name__ == "__main__":
    main()
