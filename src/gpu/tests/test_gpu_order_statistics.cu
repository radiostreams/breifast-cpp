#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "gpu_kernels.hpp"
#include "gpu_test_utils.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_median;
using breifast::gpu::gpu_median_abs;
using breifast::gpu::Stream;
using breifast::gpu::test::download_scalar;
using breifast::gpu::test::upload;
using Catch::Matchers::WithinRel;

namespace
{
float run_median(const std::vector<float> &values, bool abs_mode, float abs_scale = 1.F)
{
  const auto device_values = upload(values);
  DeviceVector<float> device_median(1);

  const Stream stream;
  if (abs_mode)
  {
    gpu_median_abs(device_values, values.size(), device_median, abs_scale, stream.get());
  }
  else
  {
    gpu_median(device_values, values.size(), device_median, stream.get());
  }
  return download_scalar(device_median.data(), stream);
}

// Same LCG as the Python reference and the CPU test, so both sides build bit-identical float32 inputs. Every value is an
// integer in [-10000, 10000] divided by 64, which is exact in float32, and the 20001-value range forces many ties.
std::vector<float> lcg_values(size_t n, uint32_t seed)
{
  std::vector<float> values(n);
  uint32_t state = seed;
  for (auto &value : values)
  {
    state = state * 1664525U + 1013904223U;
    value = static_cast<float>(static_cast<int>((state >> 8) % 20001) - 10000) / 64.F;
  }
  return values;
}
} // namespace

TEST_CASE("gpu_median and gpu_median_abs match the CPU oracle on small odd and even arrays", "[gpu][order_statistics]")
{
  CHECK_THAT(run_median({5.F, 1.F, 4.F, 2.F, 3.F}, false), WithinRel(3.F));
  CHECK_THAT(run_median({5.F, 1.F, 4.F, 2.F, 3.F, 6.F}, false), WithinRel(3.5F));
  CHECK_THAT(run_median({7.F}, false), WithinRel(7.F));
  CHECK_THAT(run_median({2.F, 8.F}, false), WithinRel(5.F));
  CHECK_THAT(run_median({-3.F, -1.F, -2.F}, false), WithinRel(-2.F));

  CHECK_THAT(run_median({-5.F, 1.F, -4.F, 2.F, 3.F}, true), WithinRel(3.F));
  CHECK_THAT(run_median({-6.F, 1.F, -4.F, 2.F, 3.F, 5.F}, true), WithinRel(3.5F));
  CHECK_THAT(run_median({-7.F}, true), WithinRel(7.F));
  CHECK_THAT(run_median({-2.F, 8.F}, true), WithinRel(5.F));
}

TEST_CASE("gpu_median averages the middle ranks when the upper rank starts a repeated run", "[gpu][order_statistics]")
{
  // Sorted {5, 7, 9, 9}: rank n/2 = 2 is the first 9, so rank 1 is 7 and the median is 8, not 9.
  CHECK_THAT(run_median({5.F, 7.F, 9.F, 9.F}, false), WithinRel(8.F));
  CHECK_THAT(run_median({5.F, 7.F, 9.F, 9.F}, true), WithinRel(8.F));
}

TEST_CASE("gpu_median matches numpy on seeded arrays with many ties, odd and even n", "[gpu][order_statistics]")
{
  // Expected values from np.median(x) and np.median(np.abs(x)) on the float32 arrays built by the same LCG.
  struct Case
  {
    size_t n;
    uint32_t seed;
    float median;
    float median_abs;
  };
  const std::vector<Case> cases{
      {1001, 7, -7.71875F, 76.09375F},
      {1000, 7, -7.5625F, 75.984375F},
      {100000, 42, 0.15625F, 78.578125F},
      {99999, 42, 0.15625F, 78.578125F},
  };

  for (const auto &c : cases)
  {
    INFO("n=" << c.n << " seed=" << c.seed);
    const auto values = lcg_values(c.n, c.seed);
    CHECK_THAT(run_median(values, false), WithinRel(c.median));
    CHECK_THAT(run_median(values, true), WithinRel(c.median_abs));
  }
}

TEST_CASE("gpu_median handles ties, zeros of both signs, subnormals and infinities like numpy",
          "[gpu][order_statistics]")
{
  constexpr float inf = std::numeric_limits<float>::infinity();
  const float subnormal_1 = 1e-40F;
  const float subnormal_2 = 2e-40F;
  const float subnormal_3 = 3e-40F;

  CHECK_THAT(run_median({4.F, 4.F, 4.F, 4.F, 4.F, 4.F, 4.F}, false), WithinRel(4.F));
  CHECK_THAT(run_median({3.F, 3.F, 3.F, 1.F, 3.F, 3.F}, false), WithinRel(3.F));
  // -0 and +0 compare equal, so this pins the value and leaves the sign of the returned zero free.
  CHECK_THAT(run_median({-0.F, 0.F, -0.F, 0.F, 0.F}, false), WithinRel(0.F));
  CHECK_THAT(run_median({subnormal_1, subnormal_3, subnormal_2}, false), WithinRel(subnormal_2));

  CHECK_THAT(run_median({-inf, 1.F, inf}, false), WithinRel(1.F));
  CHECK_THAT(run_median({1.F, inf}, false), WithinRel(inf));
  CHECK(std::isnan(run_median({-inf, inf}, false))); // (-inf + inf) / 2, same as numpy
  CHECK_THAT(run_median({-inf, 1.F, inf}, true), WithinRel(inf));
  CHECK_THAT(run_median({-inf, inf}, true), WithinRel(inf));
  CHECK_THAT(run_median({-inf, 1.F, 2.F}, true), WithinRel(2.F));
  CHECK_THAT(run_median({-0.F, 0.F, -0.F}, true), WithinRel(0.F));
}

TEST_CASE("gpu_median_abs applies its scale on the final write", "[gpu][order_statistics]")
{
  CHECK_THAT(run_median({-2.F, 4.F, -6.F}, true, 1.4826F), WithinRel(4.F * 1.4826F));
  CHECK_THAT(run_median({-2.F, 4.F, -6.F, -8.F}, true, 0.5F), WithinRel(5.F * 0.5F));
}
