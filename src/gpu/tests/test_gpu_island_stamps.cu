#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "device_vector.hpp"
#include "gpu_kernels.hpp"
#include "gpu_runtime.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_donut_median_abs;
using breifast::gpu::gpu_donut_std;
using breifast::gpu::gpu_stamp_abs_extreme;
using breifast::gpu::gpu_stamp_beam_chi2;
using breifast::gpu::gpu_stamp_extract;
using breifast::gpu::gpu_stamp_extract_no_zero;
using Catch::Matchers::Equals;
using Catch::Matchers::WithinRel;

namespace
{
constexpr float lmad_scale = 1.4826F;

std::vector<float> run_donut(const std::vector<std::vector<float>> &stamps, float scale)
{
  const size_t n_stamps = stamps.size();
  const size_t stamp_pixels = stamps.front().size();

  std::vector<float> flat;
  flat.reserve(n_stamps * stamp_pixels);
  for (const auto &stamp : stamps)
  {
    flat.insert(flat.end(), stamp.begin(), stamp.end());
  }

  std::vector<float> host_stamps(flat.begin(), flat.end());
  DeviceVector<float> device_stamps(flat.size());
  DeviceVector<float> device_out(n_stamps);
  DeviceVector<unsigned int> device_count(1, static_cast<unsigned int>(n_stamps));

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(device_stamps.data(), host_stamps.data(),
                             flat.size() * sizeof(float), stream);
  gpu_donut_median_abs(device_stamps.data(),
                       device_count.data(), static_cast<unsigned int>(n_stamps),
                       stamp_pixels, scale, device_out.data(), stream);

  std::vector<float> host_out(n_stamps);
  gpuMemcpyDeviceToHostAsync(host_out.data(), device_out.data(),
                             n_stamps * sizeof(float), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  return std::vector<float>(host_out.begin(), host_out.end());
}

float run_donut(const std::vector<float> &stamp, float scale)
{
  return run_donut(std::vector<std::vector<float>>{stamp}, scale)[0];
}

// Independent host oracle: sort the absolute values of the nonzero pixels and take the middle rank, averaged with rank
// m/2 - 1 for even m, times scale.
float host_lmad(const std::vector<float> &stamp, float scale)
{
  std::vector<float> nonzero;
  for (const float value : stamp)
  {
    if (value != 0.F)
    {
      nonzero.push_back(std::fabs(value));
    }
  }
  if (nonzero.empty())
  {
    return 0.F;
  }
  std::sort(nonzero.begin(), nonzero.end());
  const size_t m = nonzero.size();
  const float hi = nonzero[m / 2];
  const float lo = nonzero[m / 2 - 1];
  return scale * (m % 2 == 1 ? hi : (lo + hi) / 2.F);
}

// plane(y, x) = 100 + 10 * y + x, so every pixel is nonzero and the zero fill is easy to see. 6 rows, 7 columns.
const std::vector<float> stamp_plane = {
    100.F, 101.F, 102.F, 103.F, 104.F, 105.F, 106.F, 110.F, 111.F, 112.F, 113.F, 114.F, 115.F, 116.F,
    120.F, 121.F, 122.F, 123.F, 124.F, 125.F, 126.F, 130.F, 131.F, 132.F, 133.F, 134.F, 135.F, 136.F,
    140.F, 141.F, 142.F, 143.F, 144.F, 145.F, 146.F, 150.F, 151.F, 152.F, 153.F, 154.F, 155.F, 156.F,
};
constexpr size_t stamp_plane_n_y = 6;
constexpr size_t stamp_plane_n_x = 7;

// 21x21 stamp from the LCG, with the central (2 * inner_half + 1)^2 block zeroed.
std::vector<float> lcg_stamp(uint32_t seed, long inner_half)
{
  constexpr size_t side = 21;
  constexpr long radius = 10;
  std::vector<float> stamp(side * side);
  uint32_t state = seed;
  for (auto &value : stamp)
  {
    state = state * 1664525U + 1013904223U;
    value = static_cast<float>(static_cast<int>((state >> 8) % 2001) - 1000) / 8.F;
  }
  for (long row = radius - inner_half; row <= radius + inner_half; ++row)
  {
    for (long column = radius - inner_half; column <= radius + inner_half; ++column)
    {
      stamp[row * side + column] = 0.F;
    }
  }
  return stamp;
}

std::vector<float> run_abs_extreme(const std::vector<std::vector<float>> &stamps)
{
  const size_t n_stamps = stamps.size();
  const size_t stamp_pixels = stamps.front().size();

  std::vector<float> flat;
  flat.reserve(n_stamps * stamp_pixels);
  for (const auto &stamp : stamps)
  {
    flat.insert(flat.end(), stamp.begin(), stamp.end());
  }

  std::vector<float> host_stamps(flat.begin(), flat.end());
  DeviceVector<float> device_stamps(flat.size());
  DeviceVector<float> device_out(n_stamps);
  DeviceVector<unsigned int> device_count(1, static_cast<unsigned int>(n_stamps));

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(device_stamps.data(), host_stamps.data(),
                             flat.size() * sizeof(float), stream);
  gpu_stamp_abs_extreme(device_stamps.data(),
                        device_count.data(), static_cast<unsigned int>(n_stamps),
                        stamp_pixels, device_out.data(), stream);

  std::vector<float> host_out(n_stamps);
  gpuMemcpyDeviceToHostAsync(host_out.data(), device_out.data(),
                             n_stamps * sizeof(float), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  return std::vector<float>(host_out.begin(), host_out.end());
}

float run_abs_extreme(const std::vector<float> &stamp)
{
  return run_abs_extreme(std::vector<std::vector<float>>{stamp})[0];
}

std::vector<float> run_stamp_extract(const std::vector<float> &plane, size_t n_y, size_t n_x,
                                     const std::vector<long> &centre_y, const std::vector<long> &centre_x, long radius,
                                     long inner_boxsize)
{
  const size_t n_stamps = centre_y.size();
  const size_t side = static_cast<size_t>(2 * radius + 1);
  const size_t stamp_pixels = side * side;

  std::vector<float> host_plane(plane.begin(), plane.end());
  DeviceVector<float> device_plane(plane.size());
  std::vector<long> host_cy(centre_y.begin(), centre_y.end());
  DeviceVector<long> device_cy(n_stamps);
  std::vector<long> host_cx(centre_x.begin(), centre_x.end());
  DeviceVector<long> device_cx(n_stamps);
  DeviceVector<float> device_stamps(n_stamps * stamp_pixels);
  DeviceVector<unsigned int> device_count(1, static_cast<unsigned int>(n_stamps));

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(device_plane.data(), host_plane.data(),
                             plane.size() * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(device_cy.data(), host_cy.data(),
                             n_stamps * sizeof(long), stream);
  gpuMemcpyHostToDeviceAsync(device_cx.data(), host_cx.data(),
                             n_stamps * sizeof(long), stream);

  gpu_stamp_extract(device_plane.data(), n_y, n_x,
                    device_cy.data(), device_cx.data(),
                    device_count.data(), static_cast<unsigned int>(n_stamps), radius,
                    inner_boxsize, device_stamps.data(), stream);

  std::vector<float> host_stamps(n_stamps * stamp_pixels);
  gpuMemcpyDeviceToHostAsync(host_stamps.data(), device_stamps.data(),
                             n_stamps * stamp_pixels * sizeof(float), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  return std::vector<float>(host_stamps.begin(), host_stamps.end());
}

std::vector<float> run_stamp_extract_no_zero(const std::vector<float> &plane, size_t n_y, size_t n_x,
                                             const std::vector<long> &centre_y, const std::vector<long> &centre_x,
                                             long radius)
{
  const size_t n_stamps = centre_y.size();
  const size_t side = static_cast<size_t>(2 * radius + 1);
  const size_t stamp_pixels = side * side;

  std::vector<float> host_plane(plane.begin(), plane.end());
  DeviceVector<float> device_plane(plane.size());
  std::vector<long> host_cy(centre_y.begin(), centre_y.end());
  DeviceVector<long> device_cy(n_stamps);
  std::vector<long> host_cx(centre_x.begin(), centre_x.end());
  DeviceVector<long> device_cx(n_stamps);
  DeviceVector<float> device_stamps(n_stamps * stamp_pixels);
  DeviceVector<unsigned int> device_count(1, static_cast<unsigned int>(n_stamps));

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(device_plane.data(), host_plane.data(),
                             plane.size() * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(device_cy.data(), host_cy.data(),
                             n_stamps * sizeof(long), stream);
  gpuMemcpyHostToDeviceAsync(device_cx.data(), host_cx.data(),
                             n_stamps * sizeof(long), stream);

  gpu_stamp_extract_no_zero(device_plane.data(), n_y, n_x,
                            device_cy.data(), device_cx.data(),
                            device_count.data(), static_cast<unsigned int>(n_stamps),
                            radius, device_stamps.data(), stream);

  std::vector<float> host_stamps(n_stamps * stamp_pixels);
  gpuMemcpyDeviceToHostAsync(host_stamps.data(), device_stamps.data(),
                             n_stamps * stamp_pixels * sizeof(float), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  return std::vector<float>(host_stamps.begin(), host_stamps.end());
}

std::vector<float> run_donut_std(const std::vector<std::vector<float>> &stamps)
{
  const size_t n_stamps = stamps.size();
  const size_t stamp_pixels = stamps.front().size();

  std::vector<float> flat;
  flat.reserve(n_stamps * stamp_pixels);
  for (const auto &stamp : stamps)
  {
    flat.insert(flat.end(), stamp.begin(), stamp.end());
  }

  std::vector<float> host_stamps(flat.begin(), flat.end());
  DeviceVector<float> device_stamps(flat.size());
  DeviceVector<float> device_out(n_stamps);
  DeviceVector<unsigned int> device_count(1, static_cast<unsigned int>(n_stamps));

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(device_stamps.data(), host_stamps.data(),
                             flat.size() * sizeof(float), stream);
  gpu_donut_std(device_stamps.data(),
                device_count.data(), static_cast<unsigned int>(n_stamps),
                stamp_pixels, device_out.data(), stream);

  std::vector<float> host_out(n_stamps);
  gpuMemcpyDeviceToHostAsync(host_out.data(), device_out.data(),
                             n_stamps * sizeof(float), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  return std::vector<float>(host_out.begin(), host_out.end());
}

float run_donut_std(const std::vector<float> &stamp)
{
  return run_donut_std(std::vector<std::vector<float>>{stamp})[0];
}

// Independent host oracle: population standard deviation of the nonzero pixels.
double host_std(const std::vector<float> &stamp)
{
  double sum = 0.;
  double sumsq = 0.;
  size_t m = 0;
  for (const float value : stamp)
  {
    if (value != 0.F)
    {
      sum += value;
      sumsq += static_cast<double>(value) * static_cast<double>(value);
      ++m;
    }
  }
  if (m == 0)
  {
    return 0.;
  }
  const double mean = sum / static_cast<double>(m);
  return std::sqrt(sumsq / static_cast<double>(m) - mean * mean);
}

std::vector<float> run_beam_chi2(const std::vector<std::vector<float>> &stamps, const std::vector<float> &kernel,
                                 const std::vector<float> &adjacent_noise, float inv_max_to_std_ratio = 1.F)
{
  const size_t n_stamps = stamps.size();
  const size_t stamp_pixels = stamps.front().size();

  std::vector<float> flat;
  flat.reserve(n_stamps * stamp_pixels);
  for (const auto &stamp : stamps)
  {
    flat.insert(flat.end(), stamp.begin(), stamp.end());
  }

  std::vector<float> extreme(adjacent_noise.size());
  for (size_t i = 0; i < extreme.size(); ++i)
  {
    extreme[i] = adjacent_noise[i] / inv_max_to_std_ratio;
  }

  std::vector<float> host_stamps(flat.begin(), flat.end());
  DeviceVector<float> device_stamps(flat.size());
  std::vector<float> host_kernel(kernel.begin(), kernel.end());
  DeviceVector<float> device_kernel(kernel.size());
  std::vector<float> host_extreme(extreme.begin(), extreme.end());
  DeviceVector<float> device_extreme(extreme.size());
  DeviceVector<unsigned int> device_count(1, static_cast<unsigned int>(n_stamps));
  DeviceVector<float> device_out(n_stamps);

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(device_stamps.data(), host_stamps.data(),
                             flat.size() * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(device_kernel.data(), host_kernel.data(),
                             kernel.size() * sizeof(float), stream);
  gpuMemcpyHostToDeviceAsync(device_extreme.data(),
                             host_extreme.data(), extreme.size() * sizeof(float), stream);
  const float kernel_sum = std::accumulate(kernel.begin(), kernel.end(), 0.F);
  gpu_stamp_beam_chi2(device_stamps.data(), device_kernel.data(),
                      kernel_sum, device_count.data(), static_cast<unsigned int>(n_stamps),
                      stamp_pixels,
                      device_extreme.data(), inv_max_to_std_ratio,
                      device_out.data(), stream);

  std::vector<float> host_out(n_stamps);
  gpuMemcpyDeviceToHostAsync(host_out.data(), device_out.data(),
                             n_stamps * sizeof(float), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  return std::vector<float>(host_out.begin(), host_out.end());
}

float run_beam_chi2(const std::vector<float> &stamp, const std::vector<float> &kernel, float adjacent_noise,
                    float inv_max_to_std_ratio = 1.F)
{
  return run_beam_chi2(std::vector<std::vector<float>>{stamp}, kernel, std::vector<float>{adjacent_noise},
                       inv_max_to_std_ratio)[0];
}

} // namespace

TEST_CASE("gpu_donut_median_abs matches numpy on a seeded 9x9 stamp with a zeroed centre", "[gpu][island_stamps]")
{
  // numpy (default_rng seed 7, normal(0, 3), float32), then the centre 3x3 block and four more pixels set to 0.
  // 68 pixels stay nonzero. Reference lmad: np.median(np.abs(nz)) * np.float32(1.4826).
  const std::vector<float> stamp = {
      0.0F,          0.8962366F,   -0.82241356F, -2.6717756F, -1.3640124F, -2.9749396F, 0.18043081F,  4.0206456F,   -1.4766196F,
      -1.8614247F,   1.4695262F,   1.0706611F,   0.31624275F, -2.7914042F, -0.087755464F, 2.0859096F,  -4.032644F,   -1.3728473F,
      -5.703668F,    -3.8686132F,  -5.525205F,   -0.7052734F, -3.8023396F, 0.81379306F,  0.47025326F, 0.0F,         -7.550279F,
      -1.6160787F,   -0.14550284F, 0.33992696F,  0.0F,        0.0F,        0.0F,         -2.4265118F, 3.1826959F,   -2.422604F,
      -0.097565114F, 2.6531696F,   -1.7508013F,  0.0F,        0.0F,        0.0F,         -3.6751676F, 0.22842069F,  4.0764704F,
      0.0F,          2.5781481F,   0.3580621F,   0.0F,        0.0F,        0.0F,         -3.5978668F, 0.22354868F,  1.7300688F,
      -0.56634635F,  2.0487309F,   -0.19955195F, 2.0017426F,  4.315568F,   -2.0269868F,  0.6094158F,  -1.3899227F,  0.38180524F,
      -3.5615835F,   -1.7379048F,  -0.58858794F, 2.6962917F,  3.435666F,   -3.9705834F,  -2.383927F,  1.9407103F,   -5.977259F,
      -1.3895096F,   -0.2918608F,  3.771045F,    0.0F,        -0.9816403F, -1.1057277F,  -0.7505862F, 4.570588F,    -1.2840748F,
  };
  REQUIRE(stamp.size() == 81);

  CHECK_THAT(run_donut(stamp, lmad_scale), WithinRel(2.5861778259277344F));
}

TEST_CASE("gpu_donut_median_abs averages the two middle ranks for an even count and handles empty and single stamps",
          "[gpu][island_stamps]")
{
  // Two nonzero pixels, -3 and 5: abs {3, 5}, median 4, so lmad is 4 * 1.4826.
  const std::vector<float> even = {0.F, -3.F, 0.F, 0.F, 0.F, 0.F, 5.F, 0.F, 0.F};
  CHECK_THAT(run_donut(even, lmad_scale), WithinRel(4.F * lmad_scale));

  // One nonzero pixel, -2.5: abs {2.5}, median 2.5.
  std::vector<float> single(25, 0.F);
  single[7] = -2.5F;
  CHECK_THAT(run_donut(single, lmad_scale), WithinRel(2.5F * lmad_scale));

  const std::vector<float> empty(25, 0.F);
  CHECK_THAT(run_donut(empty, lmad_scale), WithinRel(0.F));
}

TEST_CASE("gpu_donut_median_abs resolves rank m/2-1 when the upper rank starts a repeated run", "[gpu][island_stamps]")
{
  // Nonzero abs {5, 7, 9, 9}: rank 2 is the first 9, so the median is 8, not 9.
  const std::vector<float> stamp = {5.F, 0.F, 7.F, 9.F, 0.F, 0.F, 9.F, 0.F, 0.F};
  CHECK_THAT(run_donut(stamp, lmad_scale), WithinRel(8.F * lmad_scale));
}

TEST_CASE("gpu_donut_median_abs computes each stamp independently, one block per stamp", "[gpu][island_stamps]")
{
  const std::vector<float> a = {-5.F, 1.F, -4.F, 2.F, 3.F, 0.F, 0.F, 0.F, 0.F}; // abs {1,2,3,4,5}, median 3
  const std::vector<float> b(9, 0.F);                                            // empty
  const std::vector<float> c = {0.F, -3.F, 0.F, 0.F, 0.F, 0.F, 5.F, 0.F, 0.F};  // abs {3,5}, median 4

  const auto out = run_donut({a, b, c}, lmad_scale);

  REQUIRE(out.size() == 3);
  CHECK_THAT(out[0], WithinRel(3.F * lmad_scale));
  CHECK_THAT(out[1], WithinRel(0.F));
  CHECK_THAT(out[2], WithinRel(4.F * lmad_scale));
}

TEST_CASE("gpu_donut_median_abs covers a stamp larger than a block against a host reference", "[gpu][island_stamps]")
{
  // 21x21 = 441 pixels, larger than the 256-thread block, so the strided loops are exercised. One even and one odd
  // nonzero count, both against an independent host oracle.
  const std::vector<float> even_stamp = lcg_stamp(7U, 2); // 441 - 25 = 416 nonzero
  std::vector<float> odd_stamp = lcg_stamp(11U, 2);       // 416 nonzero, one more zeroed to make it odd
  odd_stamp[0] = 0.F;

  REQUIRE(even_stamp.size() == 441);
  REQUIRE(host_lmad(even_stamp, 1.F) != 0.F);

  const auto out = run_donut({even_stamp, odd_stamp}, lmad_scale);

  REQUIRE(out.size() == 2);
  CHECK_THAT(out[0], WithinRel(host_lmad(even_stamp, lmad_scale)));
  CHECK_THAT(out[1], WithinRel(host_lmad(odd_stamp, lmad_scale)));
}

TEST_CASE("gpu_donut_median_abs rejects zero stamps and zero stamp pixels", "[gpu][island_stamps]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  DeviceVector<float> out(1);
  DeviceVector<unsigned int> count(1, 1);
  float *out_ptr = out.data();
  const unsigned int *count_ptr = count.data();

  REQUIRE_THROWS_AS(gpu_donut_median_abs(nullptr, nullptr, 0, 9, lmad_scale, out_ptr, stream), std::runtime_error);
  REQUIRE_THROWS_AS(gpu_donut_median_abs(nullptr, count_ptr, 1, 0, lmad_scale, out_ptr, stream), std::runtime_error);

  gpuStreamDestroy(stream);
}

TEST_CASE("gpu_stamp_abs_extreme matches numpy's max(abs(min), abs(max)) on a seeded 5x5 stamp",
          "[gpu][island_stamps]")
{
  // numpy (default_rng seed 42, 2 * normal, float32): min -3.90207028, max 2.44508266, so the extreme is the minimum.
  const std::vector<float> stamp = {
      0.609434187F, -2.07996821F, 1.50090241F, 1.88112938F, -3.90207028F, -2.60435891F, 0.255680799F,
      -0.632485211F, -0.0336023159F, -1.70608783F, 1.75879598F, 1.55558383F, 0.132061392F, 2.25448251F,
      0.935018659F, -1.7185849F, 0.737501562F, -1.91776526F, 1.75690055F, -0.0998518243F, -0.369724721F,
      -1.36185908F, 2.44508266F, -0.309058964F, -0.856655657F,
  };
  REQUIRE(stamp.size() == 25);

  CHECK_THAT(run_abs_extreme(stamp), WithinRel(3.90207028F));
}

TEST_CASE("gpu_stamp_abs_extreme takes the larger of the two sides and 0 for a zero stamp", "[gpu][island_stamps]")
{
  CHECK_THAT(run_abs_extreme(std::vector<float>{-5.F, 2.F, 3.F}), WithinRel(5.F));
  CHECK_THAT(run_abs_extreme(std::vector<float>{-2.F, 1.F, 4.F}), WithinRel(4.F));
  CHECK_THAT(run_abs_extreme(std::vector<float>{-4.F, 1.F, 4.F}), WithinRel(4.F));
  CHECK_THAT(run_abs_extreme(std::vector<float>{-3.F}), WithinRel(3.F));
  CHECK_THAT(run_abs_extreme(std::vector<float>(49, 0.F)), WithinRel(0.F));
}

TEST_CASE("gpu_stamp_abs_extreme ignores NaN pixels and reports FLT_MAX when nothing else is there",
          "[gpu][island_stamps]")
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();

  CHECK_THAT(run_abs_extreme(std::vector<float>{nan, -7.F, 2.F, nan}), WithinRel(7.F));
  CHECK_THAT(run_abs_extreme(std::vector<float>(2, nan)), WithinRel(std::numeric_limits<float>::max()));
  CHECK_THAT(run_abs_extreme(std::vector<float>{1.F, -inf, 3.F}), WithinRel(inf));
  CHECK_THAT(run_abs_extreme(std::vector<float>{1.F, inf, nan}), WithinRel(inf));
}

TEST_CASE("gpu_stamp_abs_extreme computes each stamp independently, one block per stamp", "[gpu][island_stamps]")
{
  const std::vector<float> a = {-5.F, 2.F, 3.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F}; // -> 5
  const std::vector<float> b(9, 0.F);                                          // -> 0
  const std::vector<float> c = {-1.F, 1.F, 8.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F}; // -> 8

  const auto out = run_abs_extreme({a, b, c});

  REQUIRE(out.size() == 3);
  CHECK_THAT(out[0], WithinRel(5.F));
  CHECK_THAT(out[1], WithinRel(0.F));
  CHECK_THAT(out[2], WithinRel(8.F));
}

TEST_CASE("gpu_stamp_abs_extreme covers a stamp larger than a block against a host reference",
          "[gpu][island_stamps]")
{
  // 21x21 = 441 pixels, larger than the 256-thread block, so the strided loop and the tree reduction both run.
  const std::vector<float> stamp = lcg_stamp(7U, 0);
  REQUIRE(stamp.size() == 441);

  float host_min = std::numeric_limits<float>::max();
  float host_max = std::numeric_limits<float>::lowest();
  for (const float value : stamp)
  {
    host_min = std::min(host_min, value);
    host_max = std::max(host_max, value);
  }
  const float expected = std::max(std::fabs(host_min), std::fabs(host_max));

  CHECK_THAT(run_abs_extreme(stamp), WithinRel(expected));
}

TEST_CASE("gpu_stamp_abs_extreme rejects zero stamps and zero stamp pixels", "[gpu][island_stamps]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  DeviceVector<float> out(1);
  DeviceVector<unsigned int> count(1, 1);
  float *out_ptr = out.data();
  const unsigned int *count_ptr = count.data();

  REQUIRE_THROWS_AS(gpu_stamp_abs_extreme(nullptr, nullptr, 0, 9, out_ptr, stream), std::runtime_error);
  REQUIRE_THROWS_AS(gpu_stamp_abs_extreme(nullptr, count_ptr, 1, 0, out_ptr, stream), std::runtime_error);

  gpuStreamDestroy(stream);
}

TEST_CASE("gpu_stamp_extract copies the window and zeros the inner box on a 7x7 plane", "[gpu][island_stamps]")
{
  std::vector<float> plane(49);
  for (size_t i = 0; i < plane.size(); ++i)
  {
    plane[i] = static_cast<float>(i + 1);
  }

  const auto stamps = run_stamp_extract(plane, 7, 7, {3L}, {3L}, 3, 1);

  // numpy: a[2:5, 2:5] = 0 for inner_boxsize 1, the 3x3 block around the centre pixel.
  const std::vector<float> expected = {
      1.F,  2.F,  3.F,  4.F,  5.F,  6.F,  7.F,  8.F,  9.F,  10.F, 11.F, 12.F, 13.F, 14.F,
      15.F, 16.F, 0.F,  0.F,  0.F,  20.F, 21.F, 22.F, 23.F, 0.F,  0.F,  0.F,  27.F, 28.F,
      29.F, 30.F, 0.F,  0.F,  0.F,  34.F, 35.F, 36.F, 37.F, 38.F, 39.F, 40.F, 41.F, 42.F,
      43.F, 44.F, 45.F, 46.F, 47.F, 48.F, 49.F,
  };

  REQUIRE(stamps.size() == 49);
  CHECK_THAT(stamps, Equals(expected));
}

TEST_CASE("gpu_stamp_extract zeros the cells outside the plane", "[gpu][island_stamps]")
{
  // Corner (0, 0): 16 of 25 cells fall outside the plane. The centre cell (100 in the pure-extract reference) is the
  // fused inner-box zero.
  const auto corner = run_stamp_extract(stamp_plane, stamp_plane_n_y, stamp_plane_n_x, {0L}, {0L}, 2, 0);
  const std::vector<float> expected_corner = {
      0.F,   0.F,   0.F,   0.F,   0.F,
      0.F,   0.F,   0.F,   0.F,   0.F,
      0.F,   0.F,   0.F,   101.F, 102.F,
      0.F,   0.F,   110.F, 111.F, 112.F,
      0.F,   0.F,   120.F, 121.F, 122.F,
  };
  CHECK_THAT(corner, Equals(expected_corner));

  // Opposite corner (5, 6): the window sticks out to the right and below.
  const auto far = run_stamp_extract(stamp_plane, stamp_plane_n_y, stamp_plane_n_x, {5L}, {6L}, 2, 0);
  const std::vector<float> expected_far = {
      134.F, 135.F, 136.F, 0.F, 0.F,
      144.F, 145.F, 146.F, 0.F, 0.F,
      154.F, 155.F, 0.F,   0.F, 0.F,
      0.F,   0.F,   0.F,   0.F, 0.F,
      0.F,   0.F,   0.F,   0.F, 0.F,
  };
  CHECK_THAT(far, Equals(expected_far));
}

TEST_CASE("gpu_stamp_extract clips the inner box to the radius", "[gpu][island_stamps]")
{
  // inner_boxsize 0 zeroes only the centre cell of the 5x5 window around (3, 3).
  const auto centre_only = run_stamp_extract(stamp_plane, stamp_plane_n_y, stamp_plane_n_x, {3L}, {3L}, 2, 0);
  const std::vector<float> expected_centre_only = {
      111.F, 112.F, 113.F, 114.F, 115.F,
      121.F, 122.F, 123.F, 124.F, 125.F,
      131.F, 132.F, 0.F,   134.F, 135.F,
      141.F, 142.F, 143.F, 144.F, 145.F,
      151.F, 152.F, 153.F, 154.F, 155.F,
  };
  CHECK_THAT(centre_only, Equals(expected_centre_only));

  // inner_boxsize at the radius or wider zeroes the whole stamp.
  CHECK_THAT(run_stamp_extract(stamp_plane, stamp_plane_n_y, stamp_plane_n_x, {3L}, {3L}, 2, 2),
             Equals(std::vector<float>(25, 0.F)));
  CHECK_THAT(run_stamp_extract(stamp_plane, stamp_plane_n_y, stamp_plane_n_x, {3L}, {3L}, 2, 100),
             Equals(std::vector<float>(25, 0.F)));

  // radius 0: a one-pixel stamp, the centre zeroed by any inner_boxsize.
  CHECK_THAT(run_stamp_extract(stamp_plane, stamp_plane_n_y, stamp_plane_n_x, {0L}, {0L}, 0, 0),
             Equals(std::vector<float>{0.F}));
}

TEST_CASE("gpu_stamp_extract extracts each island's stamp independently", "[gpu][island_stamps]")
{
  // Two islands on the 6x7 plane, radius 2, inner_boxsize 0, so each stamp has its centre cell zeroed.
  const auto stamps = run_stamp_extract(stamp_plane, stamp_plane_n_y, stamp_plane_n_x, {1L, 4L}, {2L, 1L}, 2, 0);

  REQUIRE(stamps.size() == 50);

  const std::vector<float> expected0 = {
      0.F,   0.F,   0.F,   0.F,   0.F,
      100.F, 101.F, 102.F, 103.F, 104.F,
      110.F, 111.F, 0.F,   113.F, 114.F,
      120.F, 121.F, 122.F, 123.F, 124.F,
      130.F, 131.F, 132.F, 133.F, 134.F,
  };
  const std::vector<float> expected1 = {
      0.F,   120.F, 121.F, 122.F, 123.F,
      0.F,   130.F, 131.F, 132.F, 133.F,
      0.F,   140.F, 0.F,   142.F, 143.F,
      0.F,   150.F, 151.F, 152.F, 153.F,
      0.F,   0.F,   0.F,   0.F,   0.F,
  };
  CHECK_THAT(std::vector<float>(stamps.begin(), stamps.begin() + 25), Equals(expected0));
  CHECK_THAT(std::vector<float>(stamps.begin() + 25, stamps.end()), Equals(expected1));
}

TEST_CASE("gpu_stamp_extract rejects zero stamps, an empty plane and negative sizes", "[gpu][island_stamps]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  DeviceVector<long> centres(1);
  DeviceVector<float> stamps(1);
  DeviceVector<unsigned int> count(1, 1);
  float *stamps_ptr = stamps.data();
  const long *centres_ptr = centres.data();
  const unsigned int *count_ptr = count.data();

  REQUIRE_THROWS_AS(gpu_stamp_extract(nullptr, 6, 7, centres_ptr, centres_ptr, nullptr, 0, 2, 0, stamps_ptr, stream),
                    std::runtime_error);
  REQUIRE_THROWS_AS(gpu_stamp_extract(nullptr, 0, 7, centres_ptr, centres_ptr, count_ptr, 1, 2, 0, stamps_ptr, stream),
                    std::runtime_error);
  REQUIRE_THROWS_AS(gpu_stamp_extract(nullptr, 6, 7, centres_ptr, centres_ptr, count_ptr, 1, -1, 0, stamps_ptr, stream),
                    std::runtime_error);
  REQUIRE_THROWS_AS(gpu_stamp_extract(nullptr, 6, 7, centres_ptr, centres_ptr, count_ptr, 1, 2, -1, stamps_ptr, stream),
                    std::runtime_error);

  gpuStreamDestroy(stream);
}

TEST_CASE("gpu_stamp_extract_no_zero copies the window without zeroing the centre", "[gpu][island_stamps]")
{
  std::vector<float> plane(49);
  for (size_t i = 0; i < plane.size(); ++i)
  {
    plane[i] = static_cast<float>(i + 1);
  }

  const auto stamps = run_stamp_extract_no_zero(plane, 7, 7, {3L}, {3L}, 3);

  std::vector<float> expected(49);
  for (size_t i = 0; i < expected.size(); ++i)
  {
    expected[i] = static_cast<float>(i + 1);
  }
  CHECK_THAT(stamps, Equals(expected));
}

TEST_CASE("gpu_stamp_extract_no_zero still zeros cells outside the plane", "[gpu][island_stamps]")
{
  const auto corner = run_stamp_extract_no_zero(stamp_plane, stamp_plane_n_y, stamp_plane_n_x, {0L}, {0L}, 2);
  const std::vector<float> expected_corner = {
      0.F,   0.F,   0.F,   0.F,   0.F,
      0.F,   0.F,   0.F,   0.F,   0.F,
      0.F,   0.F,   100.F, 101.F, 102.F,
      0.F,   0.F,   110.F, 111.F, 112.F,
      0.F,   0.F,   120.F, 121.F, 122.F,
  };
  CHECK_THAT(corner, Equals(expected_corner));
}

TEST_CASE("gpu_stamp_extract_no_zero gathers single values at radius zero", "[gpu][island_stamps]")
{
  const auto values =
      run_stamp_extract_no_zero(stamp_plane, stamp_plane_n_y, stamp_plane_n_x, {0L, 0L, 3L}, {0L, 2L, 3L}, 0);
  CHECK_THAT(values, Equals(std::vector<float>{100.F, 102.F, 133.F}));
}

TEST_CASE("gpu_stamp_extract_no_zero rejects zero stamps, an empty plane and negative radius",
          "[gpu][island_stamps]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  DeviceVector<long> centres(1);
  DeviceVector<float> stamps(1);
  float *stamps_ptr = stamps.data();
  const long *centres_ptr = centres.data();

  DeviceVector<unsigned int> count(1, 1);
  const unsigned int *count_ptr = count.data();

  REQUIRE_THROWS_AS(
      gpu_stamp_extract_no_zero(nullptr, 6, 7, centres_ptr, centres_ptr, nullptr, 0, 2, stamps_ptr, stream),
      std::runtime_error);
  REQUIRE_THROWS_AS(
      gpu_stamp_extract_no_zero(nullptr, 0, 7, centres_ptr, centres_ptr, count_ptr, 1, 2, stamps_ptr, stream),
      std::runtime_error);
  REQUIRE_THROWS_AS(
      gpu_stamp_extract_no_zero(nullptr, 6, 7, centres_ptr, centres_ptr, count_ptr, 1, -1, stamps_ptr, stream),
      std::runtime_error);

  gpuStreamDestroy(stream);
}

TEST_CASE("gpu_donut_std matches numpy's population standard deviation on the seeded 9x9 stamp",
          "[gpu][island_stamps]")
{
  // Same stamp as the donut median case; reference np.std of the 68 nonzero pixels.
  const std::vector<float> stamp = {
      0.0F,          0.8962366F,   -0.82241356F, -2.6717756F, -1.3640124F, -2.9749396F, 0.18043081F,  4.0206456F,   -1.4766196F,
      -1.8614247F,   1.4695262F,   1.0706611F,   0.31624275F, -2.7914042F, -0.087755464F, 2.0859096F,  -4.032644F,   -1.3728473F,
      -5.703668F,    -3.8686132F,  -5.525205F,   -0.7052734F, -3.8023396F, 0.81379306F,  0.47025326F, 0.0F,         -7.550279F,
      -1.6160787F,   -0.14550284F, 0.33992696F,  0.0F,        0.0F,        0.0F,         -2.4265118F, 3.1826959F,   -2.422604F,
      -0.097565114F, 2.6531696F,   -1.7508013F,  0.0F,        0.0F,        0.0F,         -3.6751676F, 0.22842069F,  4.0764704F,
      0.0F,          2.5781481F,   0.3580621F,   0.0F,        0.0F,        0.0F,         -3.5978668F, 0.22354868F,  1.7300688F,
      -0.56634635F,  2.0487309F,   -0.19955195F, 2.0017426F,  4.315568F,   -2.0269868F,  0.6094158F,  -1.3899227F,  0.38180524F,
      -3.5615835F,   -1.7379048F,  -0.58858794F, 2.6962917F,  3.435666F,   -3.9705834F,  -2.383927F,  1.9407103F,   -5.977259F,
      -1.3895096F,   -0.2918608F,  3.771045F,    0.0F,        -0.9816403F, -1.1057277F,  -0.7505862F, 4.570588F,    -1.2840748F,
  };
  REQUIRE(stamp.size() == 81);

  CHECK_THAT(run_donut_std(stamp), WithinRel(2.610630869687416, 1e-6));
}

TEST_CASE("gpu_donut_std spreads over the nonzero pixels and handles empty and single stamps",
          "[gpu][island_stamps]")
{
  // Population std of {-3, 5}: sqrt((16 + 16) / 2) = 4.
  CHECK_THAT(run_donut_std(std::vector<float>{0.F, -3.F, 0.F, 0.F, 0.F, 0.F, 5.F, 0.F, 0.F}), WithinRel(4.0));

  // A single nonzero pixel has zero spread.
  std::vector<float> single(25, 0.F);
  single[7] = -2.5F;
  CHECK_THAT(run_donut_std(single), WithinRel(0.0));

  // An all-zero stamp has no donut.
  CHECK_THAT(run_donut_std(std::vector<float>(25, 0.F)), WithinRel(0.0));
}

TEST_CASE("gpu_donut_std covers a stamp larger than a block against a host reference", "[gpu][island_stamps]")
{
  const std::vector<float> stamp = lcg_stamp(7U, 2);
  REQUIRE(stamp.size() == 441);
  CHECK_THAT(run_donut_std(stamp), WithinRel(host_std(stamp), 1e-5));
}

TEST_CASE("gpu_stamp_beam_chi2 matches numpy on a seeded 5x5 stamp against a Gaussian kernel",
          "[gpu][island_stamps]")
{
  const std::vector<float> stamp = {
      0.0F,                  0.0F,                  2.7482831478118896F,  0.3115137219429016F,  -0.16898779571056366F,
      -0.16185835003852844F, 2.898714065551758F,    4.261612892150879F,   3.0758731365203857F,  -1.4817990064620972F,
      2.083815813064575F,    3.1993911266326904F,   6.7904582023620605F,  3.159256935119629F,   0.0F,
      0.8286359906196594F,   3.153501033782959F,    4.115147590637207F,   2.176201343536377F,   1.0512244701385498F,
      -0.7413589358329773F,  -0.6925626993179321F,  1.918543815612793F,   0.0F,                 -1.7913588285446167F,
  };
  const std::vector<float> kernel = {
      0.003224542597308755F, 0.02054551988840103F, 0.038089048117399216F, 0.02054551988840103F,  0.003224542597308755F,
      0.009138146415352821F, 0.05822468921542168F, 0.10794193297624588F, 0.05822468921542168F,  0.009138146415352821F,
      0.012931677512824535F, 0.08239558339118958F, 0.15275199711322784F, 0.08239558339118958F,  0.012931677512824535F,
      0.009138146415352821F, 0.05822468921542168F, 0.10794193297624588F, 0.05822468921542168F,  0.009138146415352821F,
      0.003224542597308755F, 0.02054551988840103F, 0.038089048117399216F, 0.02054551988840103F,  0.003224542597308755F,
  };
  REQUIRE(stamp.size() == 25);
  REQUIRE(kernel.size() == 25);

  CHECK_THAT(run_beam_chi2(stamp, kernel, 1.3F), WithinRel(0.5573078192176192, 1e-5));
}

TEST_CASE("gpu_stamp_beam_chi2 scales over the whole stamp and masks the residual", "[gpu][island_stamps]")
{
  // 3x3 stamp with one nonzero pixel (4) and a flat kernel of ones: scale = 4/9, residual = 32/9, chi2 = (32/9)^2/4.
  const std::vector<float> stamp = {0.F, 0.F, 0.F, 0.F, 4.F, 0.F, 0.F, 0.F, 0.F};
  const std::vector<float> kernel(9, 1.F);

  CHECK_THAT(run_beam_chi2(stamp, kernel, 2.F), WithinRel(3.1604938271604937, 1e-6));
}

TEST_CASE("gpu_stamp_beam_chi2 computes adjacent noise on the fly", "[gpu][island_stamps]")
{
  const std::vector<float> stamp = {0.F, 0.F, 0.F, 0.F, 4.F, 0.F, 0.F, 0.F, 0.F};
  const std::vector<float> kernel(9, 1.F);

  CHECK_THAT(run_beam_chi2(stamp, kernel, 2.F, 2.F), WithinRel(3.1604938271604937, 1e-6));
}

TEST_CASE("gpu_stamp_beam_chi2 is 0 for a perfect fit, an empty stamp and a zero adjacent noise",
          "[gpu][island_stamps]")
{
  const std::vector<float> kernel(9, 1.F);

  // A flat stamp of 2 is exactly 2 * kernel: no residual.
  CHECK_THAT(run_beam_chi2(std::vector<float>(9, 2.F), kernel, 1.5F), WithinRel(0.F));

  // No nonzero pixel: the guard returns 0.
  CHECK_THAT(run_beam_chi2(std::vector<float>(9, 0.F), kernel, 1.5F), WithinRel(0.F));

  const std::vector<float> stamp = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F, 9.F};
  CHECK_THAT(run_beam_chi2(stamp, kernel, 0.F), WithinRel(0.F)); // adjacent_noise 0 -> 0 instead of a division
  CHECK(run_beam_chi2(stamp, kernel, 1.F) > 0.F);

  // Radius 0: one pixel always fits itself.
  CHECK_THAT(run_beam_chi2(std::vector<float>{3.F}, std::vector<float>{0.5F}, 1.F), WithinRel(0.F));
}

TEST_CASE("gpu_stamp_beam_chi2 rejects zero stamps and zero stamp pixels", "[gpu][island_stamps]")
{
  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  DeviceVector<float> out(1);
  DeviceVector<float> kernel(9);
  DeviceVector<float> extreme(1);
  DeviceVector<unsigned int> count(1, 1);
  float *out_ptr = out.data();
  const float *kernel_ptr = kernel.data();
  const float *extreme_ptr = extreme.data();
  const unsigned int *count_ptr = count.data();

  REQUIRE_THROWS_AS(gpu_stamp_beam_chi2(nullptr, kernel_ptr, 1.F, nullptr, 0, 9, extreme_ptr, 1.F, out_ptr, stream),
                    std::runtime_error);
  REQUIRE_THROWS_AS(gpu_stamp_beam_chi2(nullptr, kernel_ptr, 1.F, count_ptr, 1, 0, extreme_ptr, 1.F, out_ptr, stream),
                    std::runtime_error);

  gpuStreamDestroy(stream);
}
