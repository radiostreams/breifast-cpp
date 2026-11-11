#include "gpu_pipeline.hpp"
#include "gpu_runtime.hpp"

namespace breifast::gpu
{
namespace
{
constexpr size_t block_dim = 256;

__global__ void any_true_kernel(const bool *values, const size_t n, int *out_any)
{
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ bool tmp[block_dim];

  tmp[threadIdx.x] = index < n ? values[index] : false;
  __syncthreads();
  for (size_t factor = 1; factor < blockDim.x; factor *= 2)
  {
    if (threadIdx.x % (2 * factor) == 0)
    {
      tmp[threadIdx.x] = tmp[threadIdx.x] || tmp[threadIdx.x + factor];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && tmp[0])
  {
    atomicOr(out_any, 1);
  }
}
} // namespace

void gpu_any_true(const bool *values, const size_t n, int *out_any, GpuStream stream)
{
  if (n == 0)
  {
    throw std::runtime_error("gpu_any_true: empty array");
  }

  const dim3 block_size = {block_dim};
  const dim3 grid_size = {static_cast<unsigned int>((n + block_dim - 1) / block_dim)};
  gpuMemsetAsync(out_any, 0, sizeof(*out_any), stream);
  any_true_kernel<<<grid_size, block_size, 0, stream>>>(values, n, out_any);
  CHECK_GPU(cudaGetLastError());
}
} // namespace breifast::gpu