/**
 * @file beam_kernel.hpp
 * @brief Shared Gaussian beam-kernel construction for the CPU and GPU host paths.
 */
#pragma once

#include <cmath>
#include <cstddef>

#include <xtensor/containers/xtensor.hpp>

namespace breifast
{
/**
 * @brief Build a centred, rotated, unit-sum-normalized 2D anisotropic Gaussian kernel.
 *
 * Specializes `cubes.py`'s `create_multidimensional_gaussian_kernel` for the `centred=True`, odd-square-size case
 * used for beam fitting: axis offsets run `-halfsize..halfsize`, are rotated CCW by `rotate` before evaluating the
 * Gaussian, and the result is normalized to sum to 1.
 * @param halfsize Kernel half-width; full size is `2*halfsize+1` on each axis.
 * @param sigma_major Gaussian sigma along the pre-rotation first axis.
 * @param sigma_minor Gaussian sigma along the pre-rotation second axis.
 * @param rotate Counter-clockwise rotation, in radians.
 * @return `(2*halfsize+1) x (2*halfsize+1)` kernel, normalized so its elements sum to 1.
 */
inline xt::xtensor<float, 2> build_gaussian_beam_kernel(long halfsize, float sigma_major, float sigma_minor,
                                                         float rotate)
{
  const auto size = 2 * halfsize + 1;
  xt::xtensor<float, 2> kernel;
  kernel.resize({static_cast<std::size_t>(size), static_cast<std::size_t>(size)});

  const float cos_r = std::cos(rotate);
  const float sin_r = std::sin(rotate);
  float sum = 0.F;
  for (long i = 0; i < size; ++i)
  {
    const float dy = static_cast<float>(i - halfsize);
    for (long j = 0; j < size; ++j)
    {
      const float dx = static_cast<float>(j - halfsize);
      const float rot_dy = cos_r * dy - sin_r * dx;
      const float rot_dx = sin_r * dy + cos_r * dx;
      const float value = std::exp(
          -(rot_dy * rot_dy / (2.F * sigma_major * sigma_major) + rot_dx * rot_dx / (2.F * sigma_minor * sigma_minor)));
      kernel(i, j) = value;
      sum += value;
    }
  }
  kernel /= sum;
  return kernel;
}
} // namespace breifast
