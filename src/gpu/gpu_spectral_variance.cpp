#include "gpu_spectral_variance.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace breifast::gpu
{
breifast::SpectralVarianceStats assemble_spectral_variance_stats(const GpuSpectralVarianceHostResult &host, size_t n_y,
                                                                 size_t n_x, size_t n_channels)
{
  if (n_y == 0 || n_x == 0 || n_channels == 0)
  {
    throw std::runtime_error("assemble_spectral_variance_stats: zero dimension");
  }
  if (n_y > std::numeric_limits<size_t>::max() / n_x)
  {
    throw std::runtime_error("assemble_spectral_variance_stats: pixel count overflow");
  }

  const size_t n_pixels = n_y * n_x;
  if (host.sum_pow2.size() != n_pixels || host.sum_pow3.size() != n_pixels || host.sum_pow4.size() != n_pixels)
  {
    throw std::runtime_error("assemble_spectral_variance_stats: moment size mismatch");
  }
  if (host.nonzero_channels_mask.size() != n_channels)
  {
    throw std::runtime_error("assemble_spectral_variance_stats: nonzero mask size mismatch");
  }

  breifast::SpectralVarianceStats stats;
  stats.sum_pow2.resize({n_y, n_x});
  stats.sum_pow3.resize({n_y, n_x});
  stats.sum_pow4.resize({n_y, n_x});
  stats.nonzero_channels_mask.resize({n_channels});

  std::copy(host.sum_pow2.begin(), host.sum_pow2.end(), stats.sum_pow2.begin());
  std::copy(host.sum_pow3.begin(), host.sum_pow3.end(), stats.sum_pow3.begin());
  std::copy(host.sum_pow4.begin(), host.sum_pow4.end(), stats.sum_pow4.begin());
  std::transform(host.nonzero_channels_mask.begin(), host.nonzero_channels_mask.end(),
                 stats.nonzero_channels_mask.begin(), [](const unsigned char value) { return value != 0; });
  return stats;
}

breifast::SpectralVarianceStats gpu_compute_spectral_variance_stats(const float *cube, const bool *nonzero_channels,
                                                                    size_t n_y, size_t n_x, size_t n_channels,
                                                                    GpuStream stream)
{
  GpuSpectralVarianceHostResult host;
  gpu_spectral_variance_stats_host(cube, nonzero_channels, n_y, n_x, n_channels, host, stream);
  return assemble_spectral_variance_stats(host, n_y, n_x, n_channels);
}
} // namespace breifast::gpu
