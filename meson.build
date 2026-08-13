project('breifast-helpers-cpp', 'cpp',
  version : '0.1.0',
  default_options : ['cpp_std=c++23', 'buildtype=debugoptimized',
                      'cpp_args=-B/usr/bin', 'cpp_link_args=-B/usr/bin'])

py = import('python').find_installation()

pybind11_dep = dependency('pybind11')
xtl_dep = dependency('xtl')
xtensor_dep = dependency('xtensor')
xtensor_python_dep = dependency('xtensor-python')
numpy_dep = dependency('numpy')

tbb_version = '2022.3.0'
tbb_tarball_sha256 = '01598a46c1162c27253a0de0236f520fd8ee8166e9ebb84a4243574f88e6e50a'
tbb_tarball_url = 'https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v' + tbb_version + '.tar.gz'

tbb_build = custom_target('tbb-build',
  output : ['tbb-install', 'libtbb.so.12'],
  command : ['sh', '-c',
    'set -eu; ' +
    'PATH=/usr/bin:/bin:$PATH; ' +
    'src=@PRIVATE_DIR@/src; build=@PRIVATE_DIR@/build; ' +
    'mkdir -p "$src"; ' +
    'if [ ! -f "$src/tbb.tar.gz" ]; then curl -sL "' + tbb_tarball_url + '" -o "$src/tbb.tar.gz"; fi; ' +
    'echo "' + tbb_tarball_sha256 + '  $src/tbb.tar.gz" | sha256sum -c -; ' +
    'tar -xzf "$src/tbb.tar.gz" -C "$src" --strip-components=1; ' +
    'CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake -S "$src" -B "$build" ' +
    '-DCMAKE_INSTALL_PREFIX=@OUTPUT0@ -DCMAKE_BUILD_TYPE=Release ' +
    '-DTBB_TEST=OFF -DTBB_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON ' +
    '-DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc; ' +
    'cmake --build "$build" -j$(nproc); ' +
    'cmake --install "$build"; ' +
    'real="$(find @OUTPUT0@/lib* -maxdepth 1 -name "libtbb.so.12.*" ! -type l | head -1)"; ' +
    'cp "$real" @OUTPUT1@'],
  install : true,
  install_dir : [false, py.get_install_dir()],
)

tbb_install_dir = tbb_build[0].full_path()

tbb_dep = declare_dependency(
  compile_args : ['-I' + tbb_install_dir / 'include'],
  link_args : ['-L' + tbb_install_dir / 'lib64', '-L' + tbb_install_dir / 'lib',
               '-Wl,-rpath,$ORIGIN', '-ltbb'],
)

catch2_version = '3.7.1'
catch2_tarball_sha256 = 'c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c'
catch2_tarball_url = 'https://github.com/catchorg/Catch2/archive/refs/tags/v' + catch2_version + '.tar.gz'

catch2_build = custom_target('catch2-build',
  output : ['catch2-install'],
  command : ['sh', '-c',
    'set -eu; ' +
    'PATH=/usr/bin:/bin:$PATH; ' +
    'src=@PRIVATE_DIR@/src; build=@PRIVATE_DIR@/build; ' +
    'mkdir -p "$src"; ' +
    'if [ ! -f "$src/catch2.tar.gz" ]; then curl -sL "' + catch2_tarball_url + '" -o "$src/catch2.tar.gz"; fi; ' +
    'echo "' + catch2_tarball_sha256 + '  $src/catch2.tar.gz" | sha256sum -c -; ' +
    'tar -xzf "$src/catch2.tar.gz" -C "$src" --strip-components=1; ' +
    'CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake -S "$src" -B "$build" ' +
    '-DCMAKE_INSTALL_PREFIX=@OUTPUT0@ -DCMAKE_BUILD_TYPE=Release ' +
    '-DBUILD_TESTING=OFF -DCATCH_INSTALL_DOCS=OFF ' +
    '-DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc; ' +
    'cmake --build "$build" -j$(nproc); ' +
    'cmake --install "$build"'],
)

catch2_install_dir = catch2_build[0].full_path()

catch2_dep = declare_dependency(
  compile_args : ['-I' + catch2_install_dir / 'include'],
  link_args : ['-L' + catch2_install_dir / 'lib64', '-L' + catch2_install_dir / 'lib',
               '-lCatch2Main', '-lCatch2'],
)

ext_module = py.extension_module(
  'breifast_helpers',
  ['src/breifast_helpers.cpp', 'src/plane_lot_bindings.cpp', tbb_build],
  dependencies : [py.dependency(), pybind11_dep, xtl_dep, xtensor_dep, xtensor_python_dep, numpy_dep, tbb_dep, itt_dep],
  link_args : ['-static-libstdc++', '-static-libgcc'],
  install : true,
)

pytest = find_program('pytest', required : true)
test('breifast_helpers',
  pytest,
  args : [meson.project_source_root() / 'tests'],
  env : {'PYTHONPATH' : meson.current_build_dir()},
  depends : [ext_module],
)

bench = executable(
  'bench',
  ['src/bench.cpp', tbb_build],
  dependencies : [xtl_dep, xtensor_dep, tbb_dep, itt_dep],
  link_args : ['-static-libstdc++', '-static-libgcc'],
)

test_breifast_core = executable(
  'test_breifast_core',
  ['tests/cpp/test_breifast_core.cpp', tbb_build, catch2_build],
  dependencies : [xtl_dep, xtensor_dep, tbb_dep, catch2_dep, itt_dep],
  link_args : ['-static-libstdc++', '-static-libgcc'],
)
test('breifast_core', test_breifast_core)

uv = find_program('uv', required : true)
devshell_python = find_program('python3', required : true)
run_target('wheel',
  command : [uv, 'build', '--wheel', '--no-build-isolation',
             '--python', devshell_python,
             '-o', meson.project_source_root() / 'dist',
             meson.project_source_root()],
)
