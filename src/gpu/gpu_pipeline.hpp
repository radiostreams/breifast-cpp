#pragma once
#include <cstddef>
#include <sstream>
#include <string>

// Host-side interface of the GPU library: no CUDA headers, so C++23 translation units can include it.
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

/// The stream a GPU call runs on; `nullptr` is the default stream. Opaque here so the header needs no CUDA includes,
/// and implicitly a `cudaStream_t` inside CUDA translation units.
struct CUstream_st;
struct GpuStream
{
  void *handle_ = nullptr;
  GpuStream(std::nullptr_t = nullptr) {}
  GpuStream(void *h) : handle_(h) {}
#ifdef __CUDACC__
  operator ::CUstream_st *() const { return static_cast<::CUstream_st *>(handle_); }
#endif
};

/**
 * @brief OR of the accept masks of the channels `[band_start, band_stop)`, and whether any pixel ends up true.
 *
 * All pointers are device pointers. The work is enqueued on `stream` and the call returns before it finishes.
 * @param accept_mask `[n_channels, n_pixels]` bools, channel major.
 * @param n_pixels `Y * X` of one channel.
 * @param band_start First channel of the band.
 * @param band_stop One past the last channel of the band, `band_start < band_stop`.
 * @param out_mask `[n_pixels]` bools, true where any channel of the band is true.
 * @param out_any One int, nonzero when any pixel of `out_mask` is true. Written by this call, no host read-back.
 */
void gpu_mask_or(const bool *accept_mask, size_t n_pixels, int band_start, int band_stop, bool *out_mask, int *out_any,
                 GpuStream stream = nullptr);

/**
 * @brief Weighted band image: `out[p] = Σ_c cube[c,p]*weights[c]*inv_weights_sum - Σ_c mean[c,p]*inv_band_size`.
 *
 * Device pointers. `mean_image` may be `nullptr`, then the second sum is skipped. Work is enqueued on `stream`.
 * @param cube `[n_channels, n_pixels]` floats, channel major.
 * @param weights One weight per channel, indexed by the absolute channel.
 * @param mean_image `[n_channels, n_pixels]` floats, or `nullptr`.
 * @param out_plane `[n_pixels]` floats, written completely.
 */
void gpu_weighted_band_image(const float *cube, const float *weights, const float *mean_image, float *out_plane,
                             size_t n_pixels, int band_start, int band_stop, float inv_weights_sum,
                             float inv_band_size, GpuStream stream = nullptr);
} // namespace breifast::gpu
