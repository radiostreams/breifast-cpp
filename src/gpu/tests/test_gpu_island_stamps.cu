#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gpu_buffers.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_runtime.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_donut_median_abs;
using breifast::gpu::gpu_stamp_abs_extreme;
using breifast::gpu::PinnedVector;
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

  PinnedVector<float> host_stamps(flat.begin(), flat.end());
  DeviceVector<float> device_stamps(flat.size());
  DeviceVector<float> device_out(n_stamps);

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_stamps.data()), thrust::raw_pointer_cast(host_stamps.data()),
                             flat.size() * sizeof(float), stream);
  gpu_donut_median_abs(thrust::raw_pointer_cast(device_stamps.data()), n_stamps, stamp_pixels, scale,
                       thrust::raw_pointer_cast(device_out.data()), stream);

  PinnedVector<float> host_out(n_stamps);
  gpuMemcpyDeviceToHostAsync(thrust::raw_pointer_cast(host_out.data()), thrust::raw_pointer_cast(device_out.data()),
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

  PinnedVector<float> host_stamps(flat.begin(), flat.end());
  DeviceVector<float> device_stamps(flat.size());
  DeviceVector<float> device_out(n_stamps);

  cudaStream_t stream{};
  gpuStreamCreate(&stream);

  gpuMemcpyHostToDeviceAsync(thrust::raw_pointer_cast(device_stamps.data()), thrust::raw_pointer_cast(host_stamps.data()),
                             flat.size() * sizeof(float), stream);
  gpu_stamp_abs_extreme(thrust::raw_pointer_cast(device_stamps.data()), n_stamps, stamp_pixels,
                        thrust::raw_pointer_cast(device_out.data()), stream);

  PinnedVector<float> host_out(n_stamps);
  gpuMemcpyDeviceToHostAsync(thrust::raw_pointer_cast(host_out.data()), thrust::raw_pointer_cast(device_out.data()),
                             n_stamps * sizeof(float), stream);
  gpuStreamSynchronize(stream);
  gpuStreamDestroy(stream);

  return std::vector<float>(host_out.begin(), host_out.end());
}

float run_abs_extreme(const std::vector<float> &stamp)
{
  return run_abs_extreme(std::vector<std::vector<float>>{stamp})[0];
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
  float *out_ptr = thrust::raw_pointer_cast(out.data());

  REQUIRE_THROWS_AS(gpu_donut_median_abs(nullptr, 0, 9, lmad_scale, out_ptr, stream), std::runtime_error);
  REQUIRE_THROWS_AS(gpu_donut_median_abs(nullptr, 1, 0, lmad_scale, out_ptr, stream), std::runtime_error);

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
  float *out_ptr = thrust::raw_pointer_cast(out.data());

  REQUIRE_THROWS_AS(gpu_stamp_abs_extreme(nullptr, 0, 9, out_ptr, stream), std::runtime_error);
  REQUIRE_THROWS_AS(gpu_stamp_abs_extreme(nullptr, 1, 0, out_ptr, stream), std::runtime_error);

  gpuStreamDestroy(stream);
}
