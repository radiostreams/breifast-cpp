#include "gpu_beam_kernel.hpp"

#include "cpu/beam_kernel.hpp"

namespace breifast::gpu
{
std::vector<float> build_gaussian_beam_kernel_vector(long halfsize, float sigma_major, float sigma_minor, float rotate)
{
  const auto kernel = breifast::build_gaussian_beam_kernel(halfsize, sigma_major, sigma_minor, rotate);
  return std::vector<float>(kernel.begin(), kernel.end());
}
} // namespace breifast::gpu
