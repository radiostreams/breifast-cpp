# breifast-helpers-cpp

C++ (pybind11/xtensor) implementation of breifast's plane processing, exposed
to Python as `breifast_helpers`. The CPU path covers `process_plane` and the
`process_lot` batch runner (`run_batch_core`); the optional CUDA path covers
the same `process_lot` batch pipeline (`run_batch_core_gpu`).

## Prerequisites

Everything needed to build, test, and package is provided by the nix flake:

```
nix develop
```

This gives you meson, ninja, pkg-config, uv, a unified Python environment
(pip, meson-python, pybind11, numpy, xtensor-python, pytest sharing one
site-packages), and xtensor/xtl headers.

The C++ compiler is pinned to the *system* gcc/g++ (`CC`/`CXX` in the
shellHook, plus `-B/usr/bin` so g++'s own sub-tools like `ld` follow suit),
not nix's -- nix's toolchain is newer than the glibc/libstdc++ on the
actual deployment systems (the venvs this gets installed into), which would
otherwise produce a `.so` that fails to import there with
`GLIBC_x.y`/`GLIBCXX_x.y.z` errors. nix is only used as a build-time
convenience for xtensor/pybind11/etc. headers -- the resulting wheel needs
no nix involvement at runtime, and installs into any normal venv.

TBB is built from source with the system compiler too, for the same
reason: neither nix's prebuilt tbb (too new a glibc) nor the system's
`tbb-2020.3` RPM (runtime lib only, no `-devel` headers/pkgconfig
installed, and it predates oneTBB's unified API/SONAME) are usable
directly. This happens automatically as part of `meson compile` -- see the
`tbb-build` custom_target in `meson.build` -- fetching, configuring,
building, and installing oneTBB inline, no separate script involved.

## Build

CPU-only (the default; no CUDA toolkit needed):

```
meson setup builddir
meson compile -C builddir
```

With CUDA:

```
meson setup builddir-gpu -Dgpu=enabled
meson compile -C builddir-gpu
```

The GPU build needs nvcc and a CUDA 12.x toolkit. `-Dgpu_arch` selects the
device architecture and defaults to `sm_90` (H100). It links the CUDA runtime
statically and uses the system driver at runtime.

## Python API

`process_lot` accepts `use_gpu=False` (default) to select the CPU or CUDA
batch runner, and the module exposes `has_gpu` to report whether the build
contains GPU support.

- A GPU build *hard-requires* a visible CUDA device: `import breifast_helpers`
  raises `ImportError` when no device is available.
- A CPU-only build leaves `has_gpu` false and raises `RuntimeError` if
  `process_lot(..., use_gpu=True)` is called.
- The CUDA driver library is looked up through the normal loader first, then
  `/usr/lib64`, `/usr/lib/x86_64-linux-gnu`, and `/run/opengl-driver/lib`. If
  the driver lives elsewhere, add its directory to `LD_LIBRARY_PATH`.

## Test

```
meson test -C builddir -v
```

Runs the C++ unit tests and the pytest suite in `tests/` against the
just-built extension. For a GPU build, run the same command on a machine with
a CUDA device:

```
meson test -C builddir-gpu -v
```

That registers the CPU tests plus `test_gpu` (device kernels and host
assembly), `test_gpu_parity` (CPU-vs-GPU plane and batch parity), and the
pytest suite including the GPU parity test.

`tests/test_process_band_slice_fixtures.py` replays real `(input, output)` pairs captured from a live pipeline run.
See that file's docstring, and `BREIFAST_VERIFY_CPP` / `BREIFAST_DUMP_FIXTURES` / `BREIFAST_DUMP_FIXTURES_LIMIT` in `breifast.py`, for how to generate more.

Fixture `.npz` files themselves are too large to commit (hundreds of MB each) -- they live in a
public Nextcloud folder share (https://cld.chouh.fr/s/dkns3drnsmCQEPM) instead, with only their
filenames tracked in `tests/fixtures/manifest.txt`. Fetch them before running tests:

```
./scripts/fetch_fixtures.sh
```

If you've captured new fixtures locally (via `BREIFAST_DUMP_FIXTURES`) and want to add them,
upload and commit the updated manifest. This needs an `rclone` remote configured for Nextcloud's
WebDAV endpoint, set up once:

```
rclone config create nextcloud webdav url=https://cld.chouh.fr/remote.php/dav/files/<your-username>/ vendor=nextcloud user=<your-username> pass=<app-password>
```

(`<app-password>` is a Nextcloud app password, not your login password -- generate one under
Settings > Security > Devices & sessions.) Then, with `NEXTCLOUD_PATH` set to the folder backing
the public share above (`breifast-fixtures` here):

```
RCLONE_REMOTE=nextcloud NEXTCLOUD_PATH=breifast-fixtures ./scripts/upload_fixtures.sh
```

## Package (build a wheel)

CPU-only wheel:

```
meson compile wheel -C builddir
```

GPU wheel:

```
meson compile wheel-gpu -C builddir
```

Both produce `dist/breifast_helpers_cpp-0.1.0-cp3xx-cp3xx-linux_x86_64.whl`.
The GPU wheel is built with `-Dgpu=enabled` and hard-requires a CUDA device
when imported; the default wheel stays CPU-only.
