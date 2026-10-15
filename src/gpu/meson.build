# CUDA targets, included from the top-level meson.build when -Dgpu is enabled.
# nvcc takes C++20 at most, so nothing in here includes xtensor or needs C++23.

gpu_include = include_directories('.')

gpu_lib = static_library('breifast_gpu',
  [
    'gpu_pipeline.cu',
    'gpu_channel_reductions.cu',
    'gpu_pointwise.cu',
    'gpu_snr_filters.cu',
    'gpu_order_statistics.cu',
    'gpu_windows.cu',
    'gpu_islands.cu',
    'gpu_island_stamps.cu',
  ],
  include_directories : gpu_include,
)

test_gpu = executable('test_gpu',
  ['tests/test_gpu_smoke.cu', catch2_build],
  include_directories : gpu_include,
  dependencies : [catch2_dep],
  link_with : [gpu_lib],
)
test('gpu', test_gpu)
