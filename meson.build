project('breifast-helpers-cpp', 'cpp',
  version : '0.1.0',
  default_options : ['cpp_std=c++23'])

py = import('python').find_installation()

pybind11_dep = dependency('pybind11')
# xtensor.pc has no Requires: on xtl, even though xtensor headers include
# xtl ones directly, so xtl's include dir must be added explicitly
xtl_dep = dependency('xtl')
xtensor_dep = dependency('xtensor')
xtensor_python_dep = dependency('xtensor-python')
numpy_dep = dependency('numpy')
tbb_dep = dependency('tbb')

# installed as a standalone top-level module (not nested under a "breifast"
# package) so it doesn't collide with the actual breifast project's own
# top-level "breifast" package when both are installed in the same venv
py.extension_module(
  'breifast_helpers',
  'src/breifast_helpers.cpp',
  dependencies : [py.dependency(), pybind11_dep, xtl_dep, xtensor_dep, xtensor_python_dep, numpy_dep, tbb_dep],
  # statically link the C++ runtime so the built extension doesn't require a
  # libstdc++ on the target system as new as the one used to build it (nix's
  # toolchain is newer than most systems' /lib64/libstdc++.so.6)
  link_args : ['-static-libstdc++', '-static-libgcc'],
  install : true,
)

# `meson compile wheel -C builddir` (or `ninja -C builddir wheel`) builds a
# .whl into dist/. Meson itself has no notion of a wheel -- this just wraps
# the meson-python PEP 517 backend (declared in pyproject.toml) via pip, so
# it's driven the same way the rest of the build is instead of a separate
# manually-remembered command.
run_target('wheel',
  command : [py, '-m', 'pip', 'wheel', meson.project_source_root(),
             '--no-deps', '--no-build-isolation',
             '-w', meson.project_source_root() / 'dist'],
)
