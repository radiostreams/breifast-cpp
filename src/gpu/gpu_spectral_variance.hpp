#pragma once

#include <cstddef>

#include "gpu_kernels.hpp"
#include "process_plane_result.hpp"

namespace breifast::gpu
{
breifast::SpectralVarianceStats assemble_spectral_variance_stats(const GpuSpectralVarianceHostResult &host, size_t n_y,
                                                                 size_t n_x, size_t n_channels);

breifast::SpectralVarianceStats gpu_compute_spectral_variance_stats(const float *cube, const bool *nonzero_channels,
                                                                    size_t n_y, size_t n_x, size_t n_channels,
                                                                    cudaStream_t stream = nullptr);
} // namespace breifast::gpu
