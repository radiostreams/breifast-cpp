#include "gpu_band_result.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "gpu_runtime.hpp"

namespace breifast::gpu
{
void copy_band_result(const GpuBandResult &source, GpuBandHostResult &destination, GpuStream stream)
{
  if (source.label_count > source.n_pixels)
  {
    throw std::runtime_error("copy_band_result: label_count exceeds n_pixels");
  }
  if (source.histogram_bins == 0)
  {
    throw std::runtime_error("copy_band_result: histogram_bins must be greater than 0");
  }

  destination.label_count = source.label_count;
  const std::size_t count = source.label_count;
  const std::size_t bins = source.histogram_bins;

  destination.peak_value.resize(count);
  destination.peak_local_noise.resize(count);
  destination.peak_y.resize(count);
  destination.peak_x.resize(count);
  destination.island_size.resize(count);
  destination.stamp_extreme.resize(count);
  destination.donut_std.resize(count);
  destination.donut_lmad.resize(count);
  destination.beam_chi2.resize(count);
  destination.flux.resize(count);
  destination.boost.resize(count);
  destination.boost_found.resize(count);
  destination.noise_at_peak.resize(count);
  destination.rms_at_peak.resize(count);
  destination.selfnoise_at_peak.resize(count);

  destination.histogram0.resize(bins);
  destination.histogram1.resize(bins);
  destination.histogram2.resize(bins);

  if (count > 0)
  {
    gpuMemcpyDeviceToHostAsync(destination.peak_value.data(), source.peak_value, count * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(destination.peak_local_noise.data(), source.peak_local_noise, count * sizeof(float),
                               stream);
    gpuMemcpyDeviceToHostAsync(destination.peak_y.data(), source.peak_y, count * sizeof(long), stream);
    gpuMemcpyDeviceToHostAsync(destination.peak_x.data(), source.peak_x, count * sizeof(long), stream);
    gpuMemcpyDeviceToHostAsync(destination.island_size.data(), source.island_size, count * sizeof(unsigned int),
                               stream);
    gpuMemcpyDeviceToHostAsync(destination.stamp_extreme.data(), source.stamp_extreme, count * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(destination.donut_std.data(), source.donut_std, count * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(destination.donut_lmad.data(), source.donut_lmad, count * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(destination.beam_chi2.data(), source.beam_chi2, count * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(destination.flux.data(), source.flux, count * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(destination.boost.data(), source.boost, count * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(destination.boost_found.data(), source.boost_found, count * sizeof(bool), stream);
    gpuMemcpyDeviceToHostAsync(destination.noise_at_peak.data(), source.noise_at_peak, count * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(destination.rms_at_peak.data(), source.rms_at_peak, count * sizeof(float), stream);
    gpuMemcpyDeviceToHostAsync(destination.selfnoise_at_peak.data(), source.selfnoise_at_peak, count * sizeof(float),
                               stream);
  }

  gpuMemcpyDeviceToHostAsync(destination.histogram0.data(), source.histogram0, bins * sizeof(unsigned long long),
                             stream);
  gpuMemcpyDeviceToHostAsync(destination.histogram1.data(), source.histogram1, bins * sizeof(unsigned long long),
                             stream);
  gpuMemcpyDeviceToHostAsync(destination.histogram2.data(), source.histogram2, bins * sizeof(unsigned long long),
                             stream);

  gpuMemcpyDeviceToHostAsync(&destination.min_sigma, source.min_sigma, sizeof(float), stream);
  gpuMemcpyDeviceToHostAsync(&destination.max_sigma, source.max_sigma, sizeof(float), stream);
  gpuMemcpyDeviceToHostAsync(&destination.median_noise, source.median_noise, sizeof(float), stream);
  gpuMemcpyDeviceToHostAsync(&destination.rms_floor, source.rms_floor, sizeof(float), stream);
  gpuMemcpyDeviceToHostAsync(&destination.global_mad, source.global_mad, sizeof(float), stream);

  gpuStreamSynchronize(stream);
}

void copy_band_jobs(std::vector<GpuBandJob> &jobs, std::vector<GpuBandHostResult> &results)
{
  results.clear();
  results.resize(jobs.size());

  for (size_t i = 0; i < jobs.size(); ++i)
  {
    if (jobs[i].error != nullptr || jobs[i].status == GpuBandStatus::skipped)
    {
      continue;
    }
    copy_band_result(jobs[i].result, results[i], jobs[i].stream);
  }
}

void assemble_plane_bands(const std::vector<GpuBandStatus> &statuses,
                          const std::vector<GpuBandHostResult> &bands, GpuPlaneHostResult &result)
{
  if (statuses.size() != bands.size())
  {
    throw std::runtime_error("assemble_plane_bands: statuses and bands size mismatch");
  }

  result.statuses = statuses;
  result.bands = bands;
  for (auto &histogram : result.histograms)
  {
    histogram.clear();
  }
  result.median_noises.clear();
  result.min_sigma.reset();
  result.max_sigma.reset();
  bool histograms_set = false;

  for (size_t i = 0; i < statuses.size(); ++i)
  {
    if (statuses[i] == GpuBandStatus::skipped)
    {
      continue;
    }

    if (!histograms_set)
    {
      result.histograms[0] = bands[i].histogram0;
      result.histograms[1] = bands[i].histogram1;
      result.histograms[2] = bands[i].histogram2;
      histograms_set = true;
    }

    if (bands[i].min_sigma < std::numeric_limits<float>::infinity())
    {
      result.min_sigma = result.min_sigma ? std::min(*result.min_sigma, bands[i].min_sigma) : bands[i].min_sigma;
      result.max_sigma = result.max_sigma ? std::max(*result.max_sigma, bands[i].max_sigma) : bands[i].max_sigma;
    }
  }

  for (size_t i = statuses.size(); i > 0; --i)
  {
    const size_t band_index = i - 1;
    if (statuses[band_index] == GpuBandStatus::skipped)
    {
      continue;
    }
    result.median_noises.push_back(bands[band_index].median_noise);
  }
}

void gpu_process_plane(std::vector<GpuBandJob> &jobs, GpuPlaneHostResult &result)
{
  gpu_process_bands(jobs);

  std::vector<GpuBandStatus> statuses;
  statuses.reserve(jobs.size());
  for (const auto &job : jobs)
  {
    if (job.error != nullptr)
    {
      std::rethrow_exception(job.error);
    }
    statuses.push_back(job.status);
  }

  std::vector<GpuBandHostResult> bands;
  copy_band_jobs(jobs, bands);
  assemble_plane_bands(statuses, bands, result);
}
} // namespace breifast::gpu
