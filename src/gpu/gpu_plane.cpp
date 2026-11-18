#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>

#include "gpu_band.hpp"
#include "gpu_plane.hpp"

namespace breifast::gpu
{
void append_band_result(const GpuBandHostResult &band, const GpuBandLabels &labels, float max_to_std_ratio,
                        bool use_rms_noise_estimator, breifast::ProcessPlaneResult &result)
{
  const NvtxRange range{"append detections"};
  assert(max_to_std_ratio >= 0.F);

  const std::array<const std::vector<unsigned long long> *, 3> histograms{&band.histogram0, &band.histogram1,
                                                                          &band.histogram2};
  for (size_t i = 0; i < result.histograms.size(); ++i)
  {
    result.histograms[i].resize(histograms[i]->size());
    std::transform(histograms[i]->begin(), histograms[i]->end(), result.histograms[i].begin(),
                   [](const unsigned long long value) { return static_cast<long>(value); });
  }
  result.median_noises.push_back(band.median_noise);
  if (band.min_sigma < std::numeric_limits<float>::infinity())
  {
    result.min_sigma = result.min_sigma ? std::min(*result.min_sigma, band.min_sigma) : band.min_sigma;
    result.max_sigma = result.max_sigma ? std::max(*result.max_sigma, band.max_sigma) : band.max_sigma;
  }

  const size_t count = band.label_count;
  const auto &islands = band.islands;
  assert(std::apply([count](const auto &...column) { return ((column.size() >= count) && ...); }, islands.tie()));

  auto &detections = result.detections;
  detections.reserve(detections.size() + count);
  for (size_t j = 0; j < count; ++j)
  {
    const float flux = islands.flux[j];
    const float rms = islands.donut_std[j];
    const float lmad = islands.donut_lmad[j];
    const float adjacent_noise = islands.stamp_extreme[j] / max_to_std_ratio;
    const float noisemap0_at_peak =
        use_rms_noise_estimator ? std::max(islands.rms_at_peak[j], band.rms_floor) : band.global_mad;

    detections.add(islands.peak_y[j], islands.peak_x[j], static_cast<long>(islands.island_size[j]), labels.stokes_label,
                   labels.band_label, flux, islands.peak_local_noise[j],
                   islands.peak_local_noise[j] / band.median_noise, islands.peak_value[j],
                   rms != 0.F ? flux / rms : 0.F, adjacent_noise != 0.F ? flux / adjacent_noise : 0.F,
                   lmad != 0.F ? flux / lmad : 0.F, flux / band.global_mad, islands.selfnoise_at_peak[j],
                   islands.noise_at_peak[j] / noisemap0_at_peak, islands.beam_chi2[j], islands.boost[j]);
  }
}

namespace
{
/// Sum of a (2 * halfsize + 1)^2 beam kernel, with the same xtensor expression as the CPU stamp_beam_chi2, so that both
/// backends divide by the same float.
float beam_kernel_sum(const float *kernel, long halfsize)
{
  const auto side = static_cast<std::size_t>(2 * halfsize + 1);
  return xt::sum(xt::adapt(kernel, side * side, xt::no_ownership(), std::array<std::size_t, 2>{side, side}))();
}

/// Copies `array` into `buffer` on `stream`, growing it if needed, and returns the device pointer.
template <typename T, typename Array> T *upload(DeviceVector<T> &buffer, const Array &array, cudaStream_t stream)
{
  buffer.assign(array.data(), array.size());
  return buffer.data();
}

/// Same for an optional input: null when it is absent.
template <typename T, typename Array>
T *upload(DeviceVector<T> &buffer, const std::optional<Array> &array, cudaStream_t stream)
{
  return array ? upload(buffer, *array, stream) : nullptr;
}

/// `upload`, but an absent optional input empties the buffer: its data() is then null, so that a buffer kept from an
/// earlier call never passes for this call's input.
template <typename T, typename Array> void set_input(DeviceVector<T> &buffer, const Array &array)
{
  upload(buffer, array, Stream::local_stream());
}

template <typename T, typename Array> void set_input(DeviceVector<T> &buffer, const std::optional<Array> &array)
{
  if (array)
  {
    set_input(buffer, *array);
  }
  else
  {
    buffer = DeviceVector<T>{};
  }
}

/// Spectral variance moments of the slot's cube, after the bands, as the CPU compute_spectral_variance_stats computes
/// them: copied from the slot's buffers straight into the result.
SpectralVarianceStats spectral_variance_stats(GpuSlot &slot, const GpuPlaneInput &plane, const PlaneBatchItem &item)
{
  assert(plane.n_channels <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
  const std::size_t n_pixels = plane.n_pixels();
  const bool *const nonzero_channels =
      upload(slot.inputs.nonzero_channels, item.nonzero_channels, Stream::local_stream());
  slot.sum_pow2.ensure(n_pixels);
  slot.sum_pow3.ensure(n_pixels);
  slot.sum_pow4.ensure(n_pixels);
  gpu_spectral_variance_moments(plane.cube, nonzero_channels, slot.sum_pow2, slot.sum_pow3, slot.sum_pow4, n_pixels,
                                static_cast<int>(plane.n_channels), Stream::local_stream());

  SpectralVarianceStats stats;
  for (auto [host, device] : {std::pair{&stats.sum_pow2, &slot.sum_pow2}, std::pair{&stats.sum_pow3, &slot.sum_pow3},
                              std::pair{&stats.sum_pow4, &slot.sum_pow4}})
  {
    host->resize({plane.n_y, plane.n_x});
    CHECK_GPU(cudaMemcpyAsync(host->data(), device->data(), n_pixels * sizeof(float), cudaMemcpyDeviceToHost,
                              Stream::local_stream()));
  }
  Stream::local_stream().synchronize();
  stats.nonzero_channels_mask = item.nonzero_channels;
  return stats;
}
} // namespace

void GpuSharedInputs::assign(std::span<const PlaneBatchGroup> batch_groups, const PlaneConfig &config)
{
  const NvtxRange range{"upload groups"};
  groups.resize(batch_groups.size());

  const std::size_t n_bands = config.band_slices.size();
  band_any.ensure(groups.size() * n_bands);
  CHECK_GPU(cudaMemsetAsync(band_any.data(), 0, groups.size() * n_bands * sizeof(int), Stream::local_stream()));
  for (std::size_t i = 0; i < batch_groups.size(); ++i)
  {
    const auto &host = batch_groups[i];
    auto &device = groups[i];
    if (config.subtract_mean)
    {
      set_input(device.mean_image, host.mean_image);
    }
    else
    {
      device.mean_image = DeviceVector<float>{};
    }
    set_input(device.selfnoise_offset, host.selfnoise_offset);
    set_input(device.selfnoise_raw, host.selfnoise_raw);
    set_input(device.dde_noise, host.dde_noise);
    set_input(device.snr_cutoff, host.snr_cutoff);
    set_input(device.flux_cutoff, host.flux_cutoff);
    set_input(device.accept_mask, host.accept_mask);
    const std::size_t n_pixels = host.accept_mask.shape(1) * host.accept_mask.shape(2);
    device.band_masks.ensure(n_bands * n_pixels);
    for (std::size_t b = 0; b < n_bands; ++b)
    {
      const auto &slice = config.band_slices[b];
      gpu_mask_or(device.accept_mask.data(), n_pixels, slice.start, slice.stop, device.band_masks.data() + b * n_pixels,
                  band_any.data() + i * n_bands + b, Stream::local_stream());
    }
  }
  set_input(baseline_flux, config.baseline_flux);
  std::vector<int> any(groups.size() * n_bands);
  CHECK_GPU(cudaMemcpyAsync(any.data(), band_any.data(), any.size() * sizeof(int), cudaMemcpyDeviceToHost,
                            Stream::local_stream()));
  Stream::local_stream().synchronize();
  for (std::size_t i = 0; i < groups.size(); ++i)
  {
    groups[i].band_accepted.assign(any.begin() + i * n_bands, any.begin() + (i + 1) * n_bands);
  }
}

ProcessPlaneResult gpu_process_plane(GpuSlot &slot, const GpuSharedInputs &shared, const PlaneBatchItem &item,
                                     const PlaneConfig &config)
{
  const NvtxRange range{"plane"};
  const std::size_t n_channels = item.stokes_cube.shape(0);
  const std::size_t n_y = item.stokes_cube.shape(1);
  const std::size_t n_x = item.stokes_cube.shape(2);

  auto &inputs = slot.inputs;
  const auto &group_inputs = shared.groups[item.group_index];
  const GpuPlaneInput plane = [&]
  {
    const NvtxRange upload_range{"upload"};
    return GpuPlaneInput{
        .cube = upload(inputs.cube, item.stokes_cube, Stream::local_stream()),
        .weights = upload(inputs.weights, item.weights, Stream::local_stream()),
        .mean_image = group_inputs.mean_image.data(),
        .selfnoise_offset = group_inputs.selfnoise_offset.data(),
        .selfnoise_raw = group_inputs.selfnoise_raw.data(),
        .dde_noise = group_inputs.dde_noise.data(),
        .snr_cutoff = group_inputs.snr_cutoff.data(),
        .flux_cutoff = group_inputs.flux_cutoff.data(),
        .baseline_flux = shared.baseline_flux.data(),
        .n_y = n_y,
        .n_x = n_x,
        .n_channels = n_channels,
        .boxsize = config.boxsize,
        .inner_boxsize = config.inner_boxsize,
        .histogram_detail = config.histogram_detail,
        .max_sigma_bin = config.max_sigma_bin,
        .ddenoise_scaling_factor = config.ddenoise_scaling_factor,
        .reject_noise_threshold = config.reject_noise_threshold,
        .threshold = config.threshold,
        .max_to_std_ratio = config.max_to_std_ratio_value(),
        .use_rms_noise_estimator = config.use_rms_noise_estimator,
    };
  }();

  ProcessPlaneResult result;
  // Same order as the CPU process_plane: the last band first.
  const bool *const nonzero = item.nonzero_channels.data();
  for (auto slice = config.band_slices.rbegin(); slice != config.band_slices.rend(); ++slice)
  {
    const auto i_band = static_cast<std::size_t>(config.band_slices.rend() - slice - 1);
    if (group_inputs.band_accepted[i_band] == 0 ||
        std::none_of(nonzero + slice->start, nonzero + slice->stop, std::identity{}))
    {
      continue;
    }
    const std::string band_name = "band " + slice->label;
    const NvtxRange band_range{band_name};
    const auto beam = beam_kernel_for_band(item, *slice, config);
    const auto beam_halfsize = static_cast<long>(beam.halfsize);
    const GpuBand band{
        .start = slice->start,
        .stop = slice->stop,
        .selfnoise_scale_factor = slice->selfnoise_scale_factor,
        .beam_kernel = upload(inputs.beam_kernel, beam.kernel, Stream::local_stream()),
        .beam_halfsize = beam_halfsize,
        .beam_kernel_sum = beam.kernel ? beam_kernel_sum(beam.kernel->data(), beam_halfsize) : 0.F,
        .accept_mask = group_inputs.band_masks.data() + i_band * plane.n_pixels(),
    };
    gpu_process_band(plane, band, slot.workspace, slot.result, Stream::local_stream());
    copy_band_result(slot.result, slot.host, Stream::local_stream());
    append_band_result(slot.host, {item.stokes_label, slice->label}, plane.max_to_std_ratio,
                       config.use_rms_noise_estimator, result);
  }

  // After the bands, as on the CPU: the single-channel bands have written their SNR into the cube.
  if (n_channels > 1)
  {
    const NvtxRange variance_range{"variance"};
    result.variance_stats = spectral_variance_stats(slot, plane, item);
  }
  return result;
}

ProcessPlaneResult gpu_process_plane(const PlaneBatchItem &item, const PlaneBatchGroup &group,
                                     const PlaneConfig &config)
{
  GpuSlot slot;
  GpuSharedInputs shared;
  shared.assign(std::span{&group, 1}, config);
  PlaneBatchItem single_group_item = item; // copies the cube: a convenience for small planes, not for the batch path
  single_group_item.group_index = 0;
  return gpu_process_plane(slot, shared, single_group_item, config);
}
} // namespace breifast::gpu
