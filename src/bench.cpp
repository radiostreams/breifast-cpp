// Standalone executable that replays one real process_lot()/run_batch_core() call. Loads a lot dumped by breifast.py's
// BREIFAST_DUMP_LOT hook and converted to a manifest.json + raw-array directory by small_test/dump_lot_for_cpp.py.
//
// Usage: bench <manifest_dir> [num_threads] [repeat] [max_items]
//   num_threads: number of threads to bused by compute loops, <=0 means automatic (all CPUs). Default 0.
//   repeat: number of times to re-run run_batch_core().
//   max_items: keep only the first max_items planes, <=0 means all. Default 0.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

#include <ittnotify.h>
#include <nlohmann/json.hpp>
#include <oneapi/tbb/info.h>
#include <tracy/Tracy.hpp>

#include "plane_lot_dispatch.hpp"

namespace
{
using breifast::BandSlice;
using breifast::LotBatch;
using breifast::PlaneBatchGroup;
using breifast::PlaneBatchItem;
using breifast::PlaneConfig;

template <typename T, size_t N> xt::xtensor<T, N> load_tensor(const nlohmann::json &desc, const std::string &base_dir)
{
  const std::vector<size_t> shape = desc.at("shape").get<std::vector<size_t>>();
  xt::xtensor<T, N> arr;
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

template <typename T, size_t N>
std::optional<xt::xtensor<T, N>> load_optional_tensor(const nlohmann::json &j, const char *key, const std::string &base_dir)
{
  if (!j.contains(key) || j.at(key).is_null())
  {
    return std::nullopt;
  }
  return load_tensor<T, N>(j.at(key), base_dir);
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
  group.accept_mask = load_tensor<bool, 3>(j.at("accept_mask"), base_dir);
  group.selfnoise_offset = load_tensor<float, 2>(j.at("selfnoise_offset"), base_dir);
  group.selfnoise_raw = load_tensor<float, 2>(j.at("selfnoise_raw"), base_dir);
  group.dde_noise = load_tensor<float, 2>(j.at("dde_noise"), base_dir);
  group.mean_image = load_optional_tensor<float, 3>(j, "mean_image", base_dir);
  group.flux_cutoff = load_optional_tensor<float, 3>(j, "flux_cutoff", base_dir);
  group.snr_cutoff = load_optional_tensor<float, 3>(j, "snr_cutoff", base_dir);
  return group;
}

PlaneBatchItem load_item(const nlohmann::json &j, const std::string &base_dir)
{
  PlaneBatchItem item;
  item.group_index = j.at("group_index").get<size_t>();
  item.stokes_label = j.at("stokes_label").get<std::string>();
  item.stokes_cube = load_tensor<float, 3>(j.at("stokes_cube"), base_dir);
  item.weights = load_tensor<float, 1>(j.at("weights"), base_dir);
  item.nonzero_channels = load_tensor<bool, 1>(j.at("nonzero_channels"), base_dir);
  item.beam_major_axis = load_optional_tensor<float, 1>(j, "beam_major_axis", base_dir);
  item.beam_minor_axis = load_optional_tensor<float, 1>(j, "beam_minor_axis", base_dir);
  item.beam_position_angle = load_optional_tensor<float, 1>(j, "beam_position_angle", base_dir);
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
  config.fixed_beam_kernel = load_optional_tensor<double, 2>(m, "fixed_beam_kernel", base_dir);
  config.baseline_flux = load_optional_tensor<float, 2>(m, "baseline_flux", base_dir);
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

struct ProcCounters
{
  rusage usage{};
  std::map<std::string, long> vmstat;
};

ProcCounters read_proc_counters()
{
  ProcCounters counters;
  getrusage(RUSAGE_SELF, &counters.usage);
  std::ifstream vmstat("/proc/vmstat");
  std::string key;
  long value = 0;
  while (vmstat >> key >> value)
  {
    counters.vmstat[key] = value;
  }
  return counters;
}

double to_seconds(const timeval &tv)
{
  return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) * 1e-6;
}

// vmstat counters are system-wide; on an exclusive node they are effectively this process's.
void print_proc_counters_delta(const ProcCounters &before, const ProcCounters &after, double elapsed, int threads)
{
  const double user = to_seconds(after.usage.ru_utime) - to_seconds(before.usage.ru_utime);
  const double sys = to_seconds(after.usage.ru_stime) - to_seconds(before.usage.ru_stime);
  std::printf("    cpu: util=%.1f%% user=%.1fs sys=%.1fs minflt=%ld majflt=%ld nvcsw=%ld nivcsw=%ld\n",
              100.0 * (user + sys) / (elapsed * threads), user, sys,
              after.usage.ru_minflt - before.usage.ru_minflt, after.usage.ru_majflt - before.usage.ru_majflt,
              after.usage.ru_nvcsw - before.usage.ru_nvcsw, after.usage.ru_nivcsw - before.usage.ru_nivcsw);
  std::printf("    vmstat:");
  for (const char *key : {"pgfault", "thp_fault_alloc", "numa_hint_faults", "numa_hint_faults_local",
                          "numa_pages_migrated"})
  {
    const auto b = before.vmstat.find(key);
    const auto a = after.vmstat.find(key);
    if (b != before.vmstat.end() && a != after.vmstat.end())
    {
      std::printf(" %s=%ld", key, a->second - b->second);
    }
  }
  std::printf("\n");
}
} // namespace

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::fprintf(stderr, "usage: %s <manifest_dir> [num_threads] [repeat] [max_items]\n", argv[0]);
    return 1;
  }
  const std::string base_dir = argv[1];
  const int num_threads = argc > 2 ? std::atoi(argv[2]) : 0;
  const int repeat = argc > 3 ? std::atoi(argv[3]) : 1;
  const int max_items = argc > 4 ? std::atoi(argv[4]) : 0;

  std::ifstream manifest_file(base_dir + "/manifest.json");
  if (!manifest_file)
  {
    std::fprintf(stderr, "failed to open %s/manifest.json\n", base_dir.c_str());
    return 1;
  }
  nlohmann::json manifest;
  manifest_file >> manifest;

  const PlaneConfig config = load_config(manifest, base_dir);
  LotBatch base_lot = load_lot(manifest, base_dir);
  if (max_items > 0 && static_cast<size_t>(max_items) < base_lot.planes.size())
  {
    base_lot.planes.resize(max_items);
  }

  std::printf("loaded %zu planes, %zu groups from %s, num_threads=%d, repeat=%d\n", base_lot.planes.size(),
              base_lot.groups.size(), base_dir.c_str(), num_threads, repeat);
  std::printf("hardware_concurrency=%u, tbb::info::default_concurrency=%d\n", std::thread::hardware_concurrency(),
              tbb::info::default_concurrency());

  {
    const size_t n_sample = std::min<size_t>(36, base_lot.planes.size());
    std::vector<double> item_times;
    item_times.reserve(n_sample);
    breifast::ProcessBandScratch scratch;
    for (size_t i = 0; i < n_sample; ++i)
    {
      PlaneBatchItem item = base_lot.planes[i];
      const auto &group = base_lot.groups[item.group_index];
      const auto t0 = std::chrono::steady_clock::now();
      const auto result = process_plane(scratch, config, item, group);
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

  __itt_pause();

  double total = 0.0;
  for (int i = 0; i < repeat; ++i)
  {
    LotBatch lot = base_lot;
    const auto counters_before = read_proc_counters();
    const auto t0 = std::chrono::steady_clock::now();
    __itt_resume();
    const auto results = run_lot_core(lot, config, num_threads);
    __itt_pause();
    const auto t1 = std::chrono::steady_clock::now();
    const auto counters_after = read_proc_counters();
    FrameMark;
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    total += elapsed;
    size_t n_detections = 0;
    for (const auto &r : results)
    {
      n_detections += r.detections.size();
    }
    std::printf("  iteration %d: %zu items, %zu detections, compute time = %.3fs\n", i, results.size(), n_detections,
                elapsed);
    print_proc_counters_delta(counters_before, counters_after, elapsed,
                              num_threads > 0 ? num_threads : tbb::info::default_concurrency());
  }
  std::printf("run_batch_core: %d iterations, total = %.3fs, mean = %.3fs\n", repeat, total, total / repeat);

  return 0;
}
