# CUDA targets, included from the top-level meson.build when -Dgpu is enabled.
# nvcc takes C++20 at most, so nothing in here includes xtensor or needs C++23.

gpu_include = include_directories('.', '..')

gpu_lib = static_library('breifast_gpu',
  [
    'gpu_channel_reductions.cu',
    'gpu_snr_filters.cu',
    'gpu_order_statistics.cu',
    'gpu_noise_map.cu',
    'gpu_islands.cu',
    'gpu_process_band.cu',
    'gpu_island_stamps.cu',
  ],
  include_directories : gpu_include,
)

# gpu_lot_integration.cpp stays a source of its own: a program that does not call run_batch_core_gpu (test_gpu) then
# links neither TBB nor validate_plane_inputs. Its users (the extension, test_gpu_parity) link both. TBB headers only
# here; tbb_build[0], the install directory, orders the compile after TBB is built, where the whole target would
# archive its .so outputs.
gpu_host_lib = static_library('breifast_gpu_host',
  [
    'gpu_plane.cpp',
    'gpu_device.cpp',
    'gpu_lot_integration.cpp',
    tbb_build[0],
  ],
  include_directories : gpu_include,
  dependencies : [xtl_dep, xtensor_dep, xsimd_dep, cuda_headers_dep, tbb_dep.partial_dependency(compile_args : true)],
)

test_gpu = executable('test_gpu',
  ['tests/test_gpu_smoke.cu', 'tests/test_gpu_mask_or.cu', 'tests/test_gpu_weighted_band_image.cu',
   'tests/test_gpu_spectral_variance_moments.cu', 'tests/test_gpu_combine_noise_map.cu', 'tests/test_gpu_snr_filters.cu',
   'tests/test_gpu_rms_noise_map.cu', 'tests/test_gpu_order_statistics.cu', 'tests/test_gpu_island_stamps.cu',
   'tests/test_gpu_islands.cu',
   'tests/test_gpu_island_boost.cu', 'tests/test_gpu_label_stats.cu',
   'tests/test_gpu_band_workspace.cu', 'tests/test_gpu_process_band.cu',
   'tests/test_gpu_band_result.cu',
   'tests/test_gpu_plane_detections.cpp',
   'tests/test_gpu_plane_integration.cpp', catch2_build],
  include_directories : gpu_include,
  dependencies : [catch2_dep, xtl_dep, xtensor_dep, xsimd_dep, cuda_headers_dep],
  link_with : [gpu_lib, gpu_host_lib],
)
test('gpu', test_gpu)

test_gpu_parity = executable('test_gpu_parity',
  ['tests/test_gpu_parity.cpp', cpu_source_files, tbb_build, catch2_build],
  include_directories : gpu_include,
  dependencies : [catch2_dep, xtl_dep, xtensor_dep, xsimd_dep, tbb_dep, itt_dep, cuda_headers_dep],
  link_with : [gpu_lib, gpu_host_lib],
  link_language : 'cpp',
  link_args : ['-L' + cudart_libdir, '-lcudart_static', '-lrt', '-lpthread', '-ldl',
                '-Wl,-rpath,' + tbb_install_dir / 'lib64', '-Wl,-rpath,' + tbb_install_dir / 'lib'],
)
test('gpu_parity', test_gpu_parity)
