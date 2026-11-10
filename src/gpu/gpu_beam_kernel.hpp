#pragma once

#include <vector>

namespace breifast::gpu
{
std::vector<float> build_gaussian_beam_kernel_vector(long halfsize, float sigma_major, float sigma_minor,
                                                     float rotate);
}
