#include "gpu_plane_jobs.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <utility>


namespace breifast::gpu
{
GpuPlaneJobs::GpuPlaneJobs(const GpuPlaneJobSpec &spec) : GpuPlaneJobs(spec, [&spec]() {
  assert(spec.n_y > 0 && spec.n_x > 0 && spec.n_channels > 0);
  assert(spec.max_sigma_bin > 0);
  assert(spec.boxsize >= 0 && spec.inner_boxsize >= 0);
  assert(!spec.beam_size_enabled || spec.max_to_std_ratio_value > 0.F);
  assert(spec.cube != nullptr && spec.weights != nullptr && spec.nonzero_channels != nullptr &&
         spec.accept_mask != nullptr);
  assert(spec.selfnoise_offset != nullptr && spec.selfnoise_raw != nullptr && spec.dde_noise != nullptr);
  GpuPlaneSizes sizes;
  sizes.n_pixels = spec.n_y * spec.n_x;
  sizes.channel_pixels = spec.n_channels * sizes.n_pixels;
  return sizes;
}())
{
}

GpuPlaneJobs::GpuPlaneJobs(const GpuPlaneJobSpec &spec, const GpuPlaneSizes &sizes)
    : n_y(spec.n_y), n_x(spec.n_x), n_channels(spec.n_channels),
      max_to_std_ratio_value(spec.max_to_std_ratio_value),
      use_rms_noise_estimator(spec.use_rms_noise_estimator),
      cube(DeviceVector<float>::from_host(spec.cube, sizes.channel_pixels)),
      weights(DeviceVector<float>::from_host(spec.weights, spec.n_channels)),
      nonzero_channels(DeviceVector<bool>::from_host(spec.nonzero_channels, spec.n_channels)),
      accept_mask(DeviceVector<bool>::from_host(spec.accept_mask, sizes.channel_pixels)),
      selfnoise_offset(DeviceVector<float>::from_host(spec.selfnoise_offset, sizes.n_pixels)),
      selfnoise_raw(DeviceVector<float>::from_host(spec.selfnoise_raw, sizes.n_pixels)),
      dde_noise(DeviceVector<float>::from_host(spec.dde_noise, sizes.n_pixels))
{
  const std::size_t n_pixels = sizes.n_pixels;
  const std::size_t channel_pixels = sizes.channel_pixels;

  if (spec.subtract_mean && spec.mean_image != nullptr)
  {
    mean_image = DeviceVector<float>::from_host(spec.mean_image, channel_pixels);
  }
  if (spec.snr_cutoff != nullptr)
  {
    snr_cutoff = DeviceVector<float>::from_host(spec.snr_cutoff, channel_pixels);
  }
  if (spec.flux_cutoff != nullptr)
  {
    flux_cutoff = DeviceVector<float>::from_host(spec.flux_cutoff, channel_pixels);
  }
  if (spec.baseline_flux != nullptr)
  {
    baseline_flux = DeviceVector<float>::from_host(spec.baseline_flux, n_pixels);
  }

  const long radius = spec.boxsize / 2;
  const std::size_t stamp_side = static_cast<std::size_t>(2 * radius + 1);
  const std::size_t stamp_pixels = stamp_side * stamp_side;
  histogram_bins = static_cast<std::size_t>(2 * spec.max_sigma_bin);

  const bool use_fixed_beam = spec.beam_size_enabled && spec.fixed_beam_kernel != nullptr;
  if (use_fixed_beam)
  {
    assert(spec.fixed_beam_halfsize >= 0);
    const std::size_t beam_side = static_cast<std::size_t>(2 * spec.fixed_beam_halfsize + 1);
    fixed_beam_kernel = DeviceVector<float>::from_host(spec.fixed_beam_kernel, beam_side * beam_side);
  }
  const float fixed_beam_sum = use_fixed_beam ? beam_kernel_sum(spec.fixed_beam_kernel, spec.fixed_beam_halfsize) : 0.F;

  band_beam_kernels.reserve(spec.bands.size());
  band_inputs.reserve(spec.bands.size());
  band_workspace_sizes.reserve(spec.bands.size());
  labels.reserve(spec.bands.size());

  for (const auto &band : spec.bands)
  {
    assert(band.start >= 0 && band.start < band.stop && static_cast<std::size_t>(band.stop) <= spec.n_channels);

    const std::size_t band_size = static_cast<std::size_t>(band.stop - band.start);
    const float weights_sum = std::accumulate(spec.weights + band.start, spec.weights + band.stop, 0.F);
    const float inv_band_size = 1.F / static_cast<float>(band_size);

    const float *beam_device = nullptr;
    long beam_halfsize = 0;
    float beam_sum = 0.F;
    std::size_t beam_stamp_pixels = 0;
    if (spec.beam_size_enabled)
    {
      if (use_fixed_beam)
      {
        beam_device = fixed_beam_kernel->data();
        beam_halfsize = spec.fixed_beam_halfsize;
        beam_sum = fixed_beam_sum;
      }
      else
      {
        assert(spec.beam_major_axis != nullptr && spec.beam_minor_axis != nullptr &&
               spec.beam_position_angle != nullptr);

        float halfsize_accumulator = 0.F;
        for (long i_channel = band.start; i_channel < band.stop; ++i_channel)
        {
          halfsize_accumulator += spec.beam_major_axis[i_channel];
        }
        halfsize_accumulator *= 3.F / static_cast<float>(band_size);
        beam_halfsize = static_cast<long>(halfsize_accumulator) + 1;
        if (beam_halfsize < 0)
        {
          throw std::runtime_error("GpuPlaneJobs: negative Gaussian beam halfsize");
        }

        float major_weighted = 0.F;
        float minor_weighted = 0.F;
        float angle_weighted = 0.F;
        for (long i_channel = band.start; i_channel < band.stop; ++i_channel)
        {
          const float weight = spec.weights[i_channel];
          major_weighted += spec.beam_major_axis[i_channel] * weight;
          minor_weighted += spec.beam_minor_axis[i_channel] * weight;
          angle_weighted += spec.beam_position_angle[i_channel] * weight;
        }
        major_weighted /= weights_sum;
        minor_weighted /= weights_sum;
        angle_weighted /= weights_sum;

        const auto host_kernel =
            build_gaussian_beam_kernel_vector(beam_halfsize, major_weighted, minor_weighted, angle_weighted);
        band_beam_kernels.push_back(DeviceVector<float>::from_host(host_kernel.data(), host_kernel.size()));
        beam_device = band_beam_kernels.back().data();
        beam_sum = beam_kernel_sum(host_kernel.data(), beam_halfsize);
      }
      const std::size_t beam_side = static_cast<std::size_t>(2 * beam_halfsize + 1);
      beam_stamp_pixels = beam_side * beam_side;
    }

    const GpuBandWorkspaceSizes workspace_sizes{
        .n_pixels = n_pixels,
        .stamp_capacity = std::min(n_pixels, max_stamp_capacity),
        .stamp_pixels = stamp_pixels,
        .beam_stamp_pixels = beam_stamp_pixels,
        .histogram_bins = histogram_bins,
    };

    const GpuBandInput input{
        .sizes = GpuBandInputSizes{
            .cube = channel_pixels,
            .weights = spec.n_channels,
            .nonzero_channels = spec.n_channels,
            .accept_mask = channel_pixels,
            .mean_image = mean_image.has_value() ? channel_pixels : 0,
            .selfnoise_offset = n_pixels,
            .selfnoise_raw = n_pixels,
            .dde_noise = n_pixels,
            .snr_cutoff = snr_cutoff.has_value() ? channel_pixels : 0,
            .flux_cutoff = flux_cutoff.has_value() ? channel_pixels : 0,
            .baseline_flux = baseline_flux.has_value() ? n_pixels : 0,
            .beam_kernel = beam_stamp_pixels,
        },
        .cube = cube.data(),
        .weights = weights.data(),
        .nonzero_channels = nonzero_channels.data(),
        .mean_image = mean_image.has_value() ? mean_image->data() : nullptr,
        .accept_mask = accept_mask.data(),
        .selfnoise_offset = selfnoise_offset.data(),
        .selfnoise_raw = selfnoise_raw.data(),
        .dde_noise = dde_noise.data(),
        .snr_cutoff = snr_cutoff.has_value() ? snr_cutoff->data() : nullptr,
        .flux_cutoff = flux_cutoff.has_value() ? flux_cutoff->data() : nullptr,
        .baseline_flux = baseline_flux.has_value() ? baseline_flux->data() : nullptr,
        .beam_kernel = beam_device,
        .n_y = spec.n_y,
        .n_x = spec.n_x,
        .n_pixels = n_pixels,
        .n_channels = spec.n_channels,
        .band_start = band.start,
        .band_stop = band.stop,
        .boxsize = spec.boxsize,
        .inner_boxsize = spec.inner_boxsize,
        .beam_halfsize = beam_halfsize,
        .histogram_detail = spec.histogram_detail,
        .max_sigma_bin = spec.max_sigma_bin,
        .selfnoise_scale_factor = band.selfnoise_scale_factor,
        .inv_band_size = inv_band_size,
        .ddenoise_scaling_factor = spec.ddenoise_scaling_factor,
        .reject_noise_threshold = spec.reject_noise_threshold,
        .threshold = spec.threshold,
        .max_to_std_ratio_value = spec.max_to_std_ratio_value,
        .beam_kernel_sum = beam_sum,
        .use_rms_noise_estimator = spec.use_rms_noise_estimator,
        .subtract_mean = mean_image.has_value(),
    };

    band_inputs.push_back(input);
    band_workspace_sizes.push_back(workspace_sizes);
    labels.push_back({spec.stokes_label, band.label});
  }
}
} // namespace breifast::gpu
