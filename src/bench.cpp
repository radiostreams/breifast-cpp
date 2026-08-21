// Standalone executable that replays one real process_lot()/run_batch_core() call, for profiling under VTune. Loads a
// lot dumped by breifast.py's BREIFAST_DUMP_LOT hook and converted to a manifest.json + raw-array directory by
// small_test/dump_lot_for_cpp.py.
//
// Usage: bench <manifest_dir> [num_threads] [repeat]
//   num_threads: TBB arena size, <=0 means automatic (all CPUs). Default 0.
//   repeat: number of times to re-run run_batch_core(). Each iteration gets a fresh deep copy of
//           the loaded planes (process_band() mutates single-channel bands' stokes_cube in
//           place), matching the pybind11 binding's per-call fresh-copy semantics. Default 1.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <oneapi/tbb/info.h>

#include "plane_lot_dispatch.hpp"

namespace
{
using breifast::BandSlice;
using breifast::LotBatch;
using breifast::PlaneBatchGroup;
using breifast::PlaneBatchItem;
using breifast::PlaneConfig;

template <typename T> xt::xarray<T> load_array(const nlohmann::json &desc, const std::string &base_dir)
{
  const std::vector<size_t> shape = desc.at("shape").get<std::vector<size_t>>();
  xt::xarray<T> arr;
  arr.resize(shape);
  const std::string path = base_dir + "/" + desc.at("file").get<std::string>();
  std::ifstream in(path, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("failed to open " + path);
  }
  in.read(reinterpret_cast<char *>(arr.data()), static_cast<std::streamsize>(arr.size() * sizeof(T)));
  if (!in)
  {
    throw std::runtime_error("short read for " + path);
  }
  return arr;
}

template <typename T>
std::optional<xt::xarray<T>> load_optional_array(const nlohmann::json &j, const char *key, const std::string &base_dir)
{
  if (!j.contains(key) || j.at(key).is_null())
  {
    return std::nullopt;
  }
  return load_array<T>(j.at(key), base_dir);
}

BandSlice load_band_slice(const nlohmann::json &j)
{
  BandSlice band;
  band.start = j.at("start").get<long>();
  band.stop = j.at("stop").get<long>();
  band.selfnoise_scale_factor = j.at("selfnoise_scale_factor").get<float>();
  band.label = j.at("label").get<std::string>();
  return band;
}

PlaneBatchGroup load_group(const nlohmann::json &j, const std::string &base_dir)
{
  PlaneBatchGroup group;
  group.accept_mask = load_array<bool>(j.at("accept_mask"), base_dir);
  group.selfnoise_offset = load_array<float>(j.at("selfnoise_offset"), base_dir);
  group.selfnoise_raw = load_array<float>(j.at("selfnoise_raw"), base_dir);
  group.dde_noise = load_array<float>(j.at("dde_noise"), base_dir);
  group.mean_image = load_optional_array<float>(j, "mean_image", base_dir);
  group.flux_cutoff = load_optional_array<float>(j, "flux_cutoff", base_dir);
  group.snr_cutoff = load_optional_array<float>(j, "snr_cutoff", base_dir);
  return group;
}

PlaneBatchItem load_item(const nlohmann::json &j, const std::string &base_dir)
{
  PlaneBatchItem item;
  item.group_index = j.at("group_index").get<size_t>();
  item.stokes_label = j.at("stokes_label").get<std::string>();
  item.stokes_cube = load_array<float>(j.at("stokes_cube"), base_dir);
  item.weights = load_array<float>(j.at("weights"), base_dir);
  item.nonzero_channels = load_array<bool>(j.at("nonzero_channels"), base_dir);
  item.beam_major_axis = load_optional_array<float>(j, "beam_major_axis", base_dir);
  item.beam_minor_axis = load_optional_array<float>(j, "beam_minor_axis", base_dir);
  item.beam_position_angle = load_optional_array<float>(j, "beam_position_angle", base_dir);
  return item;
}

PlaneConfig load_config(const nlohmann::json &m, const std::string &base_dir)
{
  PlaneConfig config;
  config.subtract_mean = m.at("subtract_mean").get<bool>();
  config.use_rms_noise_estimator = m.at("use_rms_noise_estimator").get<bool>();
  config.boxsize = m.at("boxsize").get<long>();
  config.inner_boxsize = m.at("inner_boxsize").get<long>();
  config.ddenoise_scaling_factor = m.at("ddenoise_scaling_factor").get<float>();
  config.full_bandwidth = m.at("full_bandwidth").get<float>();
  config.reject_noise_threshold = m.at("reject_noise_threshold").get<float>();
  config.histogram_detail = m.at("histogram_detail").get<long>();
  config.max_sigma_bin = m.at("max_sigma_bin").get<long>();
  config.threshold = m.at("threshold").get<float>();
  config.save_stat_cubes = m.at("save_stat_cubes").get<bool>();
  config.beam_size_enabled = m.at("beam_size_enabled").get<bool>();
  config.fixed_beam_halfsize = m.at("fixed_beam_halfsize").get<long>();
  config.fixed_beam_kernel = load_optional_array<double>(m, "fixed_beam_kernel", base_dir);
  config.baseline_flux = load_optional_array<float>(m, "baseline_flux", base_dir);
  for (const auto &band_json : m.at("band_slices"))
  {
    config.band_slices.push_back(load_band_slice(band_json));
  }
  return config;
}

LotBatch load_lot(const nlohmann::json &m, const std::string &base_dir)
{
  LotBatch lot;
  for (size_t i = 0; i < 30; ++i)
  {
    for (const auto &group_json : m.at("groups"))
    {
      lot.groups.push_back(load_group(group_json, base_dir));
    }
    for (const auto &plane_json : m.at("planes"))
    {
      lot.planes.push_back(load_item(plane_json, base_dir));
    }
  }
  return lot;
}
} // namespace

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::fprintf(stderr, "usage: %s <manifest_dir> [num_threads] [repeat]\n", argv[0]);
    return 1;
  }
  const std::string base_dir = argv[1];
  const int num_threads = argc > 2 ? std::atoi(argv[2]) : 0;
  const int repeat = argc > 3 ? std::atoi(argv[3]) : 1;

  std::ifstream manifest_file(base_dir + "/manifest.json");
  if (!manifest_file)
  {
    std::fprintf(stderr, "failed to open %s/manifest.json\n", base_dir.c_str());
    return 1;
  }
  nlohmann::json manifest;
  manifest_file >> manifest;

  const PlaneConfig config = load_config(manifest, base_dir);
  const LotBatch base_lot = load_lot(manifest, base_dir);

  std::printf("loaded %zu planes, %zu groups from %s, num_threads=%d, repeat=%d\n", base_lot.planes.size(),
              base_lot.groups.size(), base_dir.c_str(), num_threads, repeat);
  std::printf("hardware_concurrency=%u, tbb::info::default_concurrency=%d\n", std::thread::hardware_concurrency(),
              tbb::info::default_concurrency());

  // Single-threaded per-item cost distribution over one unique pass (bench.cpp's loader
  // duplicates the manifest N times into base_lot, so timing beyond the first manifest-sized
  // chunk would just repeat the same distribution).
  {
    const size_t n_sample = std::min<size_t>(36, base_lot.planes.size());
    std::vector<double> item_times;
    item_times.reserve(n_sample);
    for (size_t i = 0; i < n_sample; ++i)
    {
      PlaneBatchItem item = base_lot.planes[i];
      const auto &group = base_lot.groups[item.group_index];
      const auto t0 = std::chrono::steady_clock::now();
      const auto result = process_plane(config, item, group);
      const auto t1 = std::chrono::steady_clock::now();
      item_times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::vector<double> sorted_times = item_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    const double sum = std::accumulate(sorted_times.begin(), sorted_times.end(), 0.0);
    std::printf("per-item cost (single-threaded, n=%zu): min=%.4fs p25=%.4fs median=%.4fs p75=%.4fs max=%.4fs "
               "mean=%.4fs\n",
               n_sample, sorted_times.front(), sorted_times[n_sample / 4], sorted_times[n_sample / 2],
               sorted_times[3 * n_sample / 4], sorted_times.back(), sum / static_cast<double>(n_sample));
  }

  double total = 0.0;
  for (int i = 0; i < repeat; ++i)
  {
    LotBatch lot = base_lot; // deep copy: process_band() mutates single-channel bands in place
    const auto t0 = std::chrono::steady_clock::now();
    const auto results = run_lot_core(lot, config, num_threads);
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    total += elapsed;
    size_t n_detections = 0;
    for (const auto &r : results)
    {
      n_detections += r.detections.size();
    }
    std::printf("  iteration %d: %zu items, %zu detections, compute time = %.3fs\n", i, results.size(), n_detections,
                elapsed);
  }
  std::printf("run_batch_core: %d iterations, total = %.3fs, mean = %.3fs\n", repeat, total, total / repeat);

  return 0;
}
