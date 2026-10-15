#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "cpu/order_statistics.hpp"

using namespace breifast::detail::median;

namespace
{
bool same_value(float a, float b)
{
  return (std::isnan(a) && std::isnan(b)) || a == b;
}

// Both helpers also run the radix version and require it to agree, so every value pinned below covers both.
float median(const std::vector<float> &values, std::vector<float> &scratch)
{
  const float nth = median_of<float>({values.data(), values.size()}, scratch);
  CHECK(same_value(radix_median_of({values.data(), values.size()}), nth));
  return nth;
}

float median_abs(const std::vector<float> &values, std::vector<float> &scratch)
{
  const float nth = median_of_abs<float>({values.data(), values.size()}, scratch);
  CHECK(same_value(radix_median_of_abs({values.data(), values.size()}), nth));
  return nth;
}
} // namespace

TEST_CASE("median_of and median_of_abs pick the middle value for odd n and average the two middle values for even n",
          "[median]")
{
  // One scratch reused across every call, shrinking and growing, as process_band does.
  std::vector<float> scratch;

  CHECK(median({5.F, 1.F, 4.F, 2.F, 3.F}, scratch) == 3.F);
  CHECK(median({5.F, 1.F, 4.F, 2.F, 3.F, 6.F}, scratch) == 3.5F);
  CHECK(median({7.F}, scratch) == 7.F);
  CHECK(median({2.F, 8.F}, scratch) == 5.F);
  CHECK(median({-3.F, -1.F, -2.F}, scratch) == -2.F);

  // Signs that would reorder the plain values, to prove the abs version ranks by magnitude.
  CHECK(median_abs({-5.F, 1.F, -4.F, 2.F, 3.F}, scratch) == 3.F);
  CHECK(median_abs({-6.F, 1.F, -4.F, 2.F, 3.F, 5.F}, scratch) == 3.5F);
  CHECK(median_abs({-7.F}, scratch) == 7.F);
  CHECK(median_abs({-2.F, 8.F}, scratch) == 5.F);
}

namespace
{
// Same LCG as the Python reference, so both sides build bit-identical float32 inputs. Every value is an integer in
// [-10000, 10000] divided by 64, which is exact in float32, and the 20001-value range forces many ties.
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

TEST_CASE("median_of and median_of_abs match numpy's median on seeded arrays with many ties, odd and even n",
          "[median]")
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

  std::vector<float> scratch;
  for (const auto &c : cases)
  {
    INFO("n=" << c.n << " seed=" << c.seed);
    const auto values = lcg_values(c.n, c.seed);
    CHECK(median(values, scratch) == c.median);
    CHECK(median_abs(values, scratch) == c.median_abs);
  }
}

TEST_CASE("median_of and median_of_abs handle ties, zeros of both signs, subnormals and infinities like numpy",
          "[median]")
{
  constexpr float inf = std::numeric_limits<float>::infinity();
  const float subnormal_1 = 1e-40F;
  const float subnormal_2 = 2e-40F;
  const float subnormal_3 = 3e-40F;
  REQUIRE(std::fpclassify(subnormal_2) == FP_SUBNORMAL);

  std::vector<float> scratch;

  CHECK(median({4.F, 4.F, 4.F, 4.F, 4.F, 4.F, 4.F}, scratch) == 4.F);
  CHECK(median({3.F, 3.F, 3.F, 1.F, 3.F, 3.F}, scratch) == 3.F);
  // -0 and +0 compare equal, so this pins the value and leaves the sign of the returned zero free.
  CHECK(median({-0.F, 0.F, -0.F, 0.F, 0.F}, scratch) == 0.F);
  CHECK(median({subnormal_1, subnormal_3, subnormal_2}, scratch) == subnormal_2);

  CHECK(median({-inf, 1.F, inf}, scratch) == 1.F);
  CHECK(median({1.F, inf}, scratch) == inf);
  CHECK(std::isnan(median({-inf, inf}, scratch))); // (-inf + inf) / 2, same as numpy
  CHECK(median_abs({-inf, 1.F, inf}, scratch) == inf);
  CHECK(median_abs({-inf, inf}, scratch) == inf);
  CHECK(median_abs({-inf, 1.F, 2.F}, scratch) == 2.F);
  CHECK(median_abs({-0.F, 0.F, -0.F}, scratch) == 0.F);
}

TEST_CASE("radix_median_of and radix_median_of_abs return the same value as the nth_element versions on random arrays",
          "[median][radix]")
{
  // Each mode stresses a different part of the radix select: bin boundaries (normal, wide dynamic range), rank k-1
  // falling in a lower bin (heavy ties, few distinct values), key ordering around zero and at the extremes.
  std::mt19937 rng(20260929);
  std::vector<float> scratch;
  size_t checked = 0;
  for (int trial = 0; trial < 6000; ++trial)
  {
    const size_t n = 1 + rng() % (trial % 10 == 0 ? 5000 : 40);
    std::vector<float> values(n);
    const int mode = trial % 6;
    for (auto &value : values)
    {
      switch (mode)
      {
      case 0:
        value = std::normal_distribution<float>(0.F, 1.F)(rng);
        break;
      case 1:
        value = static_cast<float>(static_cast<int>(rng() % 7) - 3);
        break;
      case 2:
        value = std::ldexp(static_cast<float>(rng() % 1000), -static_cast<int>(rng() % 160)); // subnormal to normal
        break;
      case 3:
        value = (rng() % 3 == 0) ? -0.F : static_cast<float>(rng() % 3);
        break;
      case 4:
        value = std::exponential_distribution<float>(1.F)(rng) * ((rng() % 2) ? 1e30F : -1e-30F);
        break;
      default:
      {
        constexpr float inf = std::numeric_limits<float>::infinity();
        const float choices[] = {-inf, inf, 1.F, -1.F, 1.00000012F, 0.99999994F};
        value = choices[rng() % 6];
      }
      }
    }
    INFO("trial=" << trial << " mode=" << mode << " n=" << n);
    const float expected = median(values, scratch);
    const float actual = radix_median_of({values.data(), values.size()});
    CHECK(same_value(actual, expected));
    CHECK(radix_median_of_abs({values.data(), values.size()}) == median_abs(values, scratch));
    ++checked;
  }
  CHECK(checked == 6000);
}
