#pragma once

#include <cstddef>
#include <sstream>
#include <string>

// Public and CUDA-free: the extension module includes it.
namespace breifast::gpu
{
struct GpuDeviceSummary
{
  std::string name;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  size_t total_global_memory = 0;
  int multiprocessor_count = 0;
  int max_threads_per_block = 0;
  int max_threads_per_multiprocessor = 0;

  std::string to_string() const
  {
    std::stringstream ss;
    ss << "GPU Device: " << name << "\n";
    ss << "  Compute capability: " << compute_capability_major << "." << compute_capability_minor << "\n";
    ss << "  Total global memory: " << (total_global_memory / (1024 * 1024)) << " MiB\n";
    ss << "  Multiprocessors: " << multiprocessor_count << "\n";
    ss << "  Max threads per block: " << max_threads_per_block << "\n";
    ss << "  Max threads per multiprocessor: " << max_threads_per_multiprocessor << "\n";
    return ss.str();
  }
};

/// Properties of the GPU with index `device_id` among the devices visible to the process.
GpuDeviceSummary gpu_device_summary(int device_id = 0);
void require_cuda_device();
} // namespace breifast::gpu
