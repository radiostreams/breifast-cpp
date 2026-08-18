#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "plane_lot_dispatch.hpp"

namespace
{
constexpr float pi = std::numbers::pi_v<float>;

struct KernelCase
{
  long halfsize;
  float sigma_major;
  float sigma_minor;
  float rotate;
};

// Halfsize edges (0 = single pixel, 1 = smallest real kernel, 10 = larger), sigma edges (very narrow, very
// wide, extreme anisotropy in both orderings), and rotate edges (0, small, axis-aligned quarter/half/full
// turns, negative angles).
const std::vector<KernelCase> kernel_cases = {
    {0, 1.0F, 1.0F, 0.F},
    {1, 1.0F, 1.0F, 0.F},
    {3, 1.5F, 1.5F, 0.F},
    {5, 2.0F, 1.0F, 0.3F},
    {2, 0.7F, 0.7F, 1.2F},
    {4, 0.05F, 0.05F, 0.F},
    {4, 0.001F, 0.001F, 0.F},
    {10, 50.F, 50.F, 0.F},
    {10, 1000.F, 1000.F, 0.F},
    {6, 100.F, 0.02F, 0.9F},
    {6, 0.02F, 100.F, 0.9F},
    {5, 2.0F, 1.0F, pi},
    {5, 2.0F, 1.0F, -pi / 4.F},
    {5, 2.0F, 1.0F, 2.F * pi},
};
} // namespace

using namespace breifast;

TEST_CASE("build_gaussian_beam_kernel normalizes to sum 1", "[build_gaussian_beam_kernel]")
{
  for (const auto &c : kernel_cases)
  {
    const auto kernel = build_gaussian_beam_kernel(c.halfsize, c.sigma_major, c.sigma_minor, c.rotate);
    CHECK(xt::sum(kernel)() == Catch::Approx(1.0).epsilon(1e-5));
  }
}

TEST_CASE("build_gaussian_beam_kernel has shape (2*halfsize+1) x (2*halfsize+1)", "[build_gaussian_beam_kernel]")
{
  for (const auto &c : kernel_cases)
  {
    const auto kernel = build_gaussian_beam_kernel(c.halfsize, c.sigma_major, c.sigma_minor, c.rotate);
    const auto expected_size = static_cast<size_t>(2 * c.halfsize + 1);
    CHECK(kernel.shape(0) == expected_size);
    CHECK(kernel.shape(1) == expected_size);
  }
}

TEST_CASE("build_gaussian_beam_kernel peaks at the center", "[build_gaussian_beam_kernel]")
{
  for (const auto &c : kernel_cases)
  {
    const auto kernel = build_gaussian_beam_kernel(c.halfsize, c.sigma_major, c.sigma_minor, c.rotate);
    const auto center = kernel(c.halfsize, c.halfsize);
    for (size_t i = 0; i < kernel.shape(0); ++i)
    {
      for (size_t j = 0; j < kernel.shape(1); ++j)
      {
        CHECK(kernel(i, j) <= center);
      }
    }
  }
}

TEST_CASE("build_gaussian_beam_kernel is rotation-invariant and transpose-symmetric when isotropic",
         "[build_gaussian_beam_kernel]")
{
  const std::vector<float> sigmas = {0.001F, 0.05F, 1.5F, 50.F, 1000.F};
  const std::vector<float> rotates = {0.F, 0.3F, pi / 2.F, pi, -pi / 3.F, 2.F * pi};
  const long halfsize = 4;

  for (const auto sigma : sigmas)
  {
    const auto reference = build_gaussian_beam_kernel(halfsize, sigma, sigma, 0.F);
    for (const auto rotate : rotates)
    {
      const auto kernel = build_gaussian_beam_kernel(halfsize, sigma, sigma, rotate);
      for (size_t i = 0; i < kernel.shape(0); ++i)
      {
        for (size_t j = 0; j < kernel.shape(1); ++j)
        {
          CHECK(kernel(i, j) == Catch::Approx(reference(i, j)).epsilon(1e-4));
          CHECK(kernel(i, j) == Catch::Approx(kernel(j, i)).epsilon(1e-4));
        }
      }
    }
  }
}

TEST_CASE("build_gaussian_beam_kernel swaps axes under a 90-degree rotation", "[build_gaussian_beam_kernel]")
{
  const std::vector<std::pair<float, float>> sigma_pairs = {{2.0F, 1.0F}, {100.F, 0.02F}, {0.02F, 100.F},
                                                             {50.F, 0.5F}};
  const std::vector<long> halfsizes = {2, 4, 6};

  for (const auto halfsize : halfsizes)
  {
    for (const auto &[sigma_major, sigma_minor] : sigma_pairs)
    {
      const auto kernel_rotated = build_gaussian_beam_kernel(halfsize, sigma_major, sigma_minor, pi / 2.F);
      const auto kernel_swapped = build_gaussian_beam_kernel(halfsize, sigma_minor, sigma_major, 0.F);

      for (size_t i = 0; i < kernel_rotated.shape(0); ++i)
      {
        for (size_t j = 0; j < kernel_rotated.shape(1); ++j)
        {
          CHECK(kernel_rotated(i, j) == Catch::Approx(kernel_swapped(i, j)).epsilon(1e-4));
        }
      }
    }
  }
}

TEST_CASE("build_gaussian_beam_kernel is periodic with period pi in rotate", "[build_gaussian_beam_kernel]")
{
  const std::vector<float> base_rotates = {0.F, 0.3F, pi / 4.F, -pi / 6.F, 1.9F};
  const long halfsize = 5;
  const float sigma_major = 2.0F;
  const float sigma_minor = 0.8F;

  for (const auto rotate : base_rotates)
  {
    const auto kernel = build_gaussian_beam_kernel(halfsize, sigma_major, sigma_minor, rotate);
    const auto kernel_plus_pi = build_gaussian_beam_kernel(halfsize, sigma_major, sigma_minor, rotate + pi);

    for (size_t i = 0; i < kernel.shape(0); ++i)
    {
      for (size_t j = 0; j < kernel.shape(1); ++j)
      {
        CHECK(kernel(i, j) == Catch::Approx(kernel_plus_pi(i, j)).epsilon(1e-4));
      }
    }
  }
}

TEST_CASE("build_gaussian_beam_kernel is unaffected by the sign of sigma", "[build_gaussian_beam_kernel]")
{
  const long halfsize = 4;
  const std::vector<std::pair<float, float>> sigma_pairs = {{1.5F, 1.5F}, {2.0F, 1.0F}, {0.05F, 3.0F}};
  const std::vector<float> rotates = {0.F, 0.5F, pi / 2.F};

  for (const auto &[sigma_major, sigma_minor] : sigma_pairs)
  {
    for (const auto rotate : rotates)
    {
      const auto kernel_positive = build_gaussian_beam_kernel(halfsize, sigma_major, sigma_minor, rotate);
      const auto kernel_negative = build_gaussian_beam_kernel(halfsize, -sigma_major, -sigma_minor, rotate);

      for (size_t i = 0; i < kernel_positive.shape(0); ++i)
      {
        for (size_t j = 0; j < kernel_positive.shape(1); ++j)
        {
          CHECK(kernel_positive(i, j) == Catch::Approx(kernel_negative(i, j)).epsilon(1e-4));
        }
      }
    }
  }
}

TEST_CASE("build_gaussian_beam_kernel matches a Python reference for an anisotropic, non-axis-aligned rotated "
         "kernel",
         "[build_gaussian_beam_kernel]")
{
  // sigma_major != sigma_minor and rotate=0.7 (not 0, pi/2, pi, or any other special angle already probed by the
  // symmetry tests above) -- this case has no self-symmetry to check against itself, so it only catches bugs like a
  // swapped sin/cos sign or a mixed-up dx/dy by comparison against an independent Python computation of the same
  // formula.
  const auto kernel = build_gaussian_beam_kernel(/*halfsize=*/3, /*sigma_major=*/2.3F, /*sigma_minor=*/0.9F,
                                                  /*rotate=*/0.7F);

  const float expected[7][7] = {
      {1.324408431e-06F, 4.597427900e-05F, 7.166182932e-04F, 5.015807104e-03F, 1.576426530e-02F, 2.224777669e-02F,
       1.409871304e-02F},
      {2.948308997e-05F, 6.114134862e-04F, 5.693469558e-03F, 2.380668715e-02F, 4.469931826e-02F, 3.768623977e-02F,
       1.426739507e-02F},
      {3.520309140e-04F, 4.361258664e-03F, 2.426179116e-02F, 6.060575249e-02F, 6.798057105e-02F, 3.424015656e-02F,
       7.744023245e-03F},
      {2.254474634e-03F, 1.668573805e-02F, 5.545305307e-02F, 8.275332938e-02F, 5.545305307e-02F, 1.668573805e-02F,
       2.254474634e-03F},
      {7.744023245e-03F, 3.424015656e-02F, 6.798057105e-02F, 6.060575249e-02F, 2.426179116e-02F, 4.361258664e-03F,
       3.520309140e-04F},
      {1.426739507e-02F, 3.768623977e-02F, 4.469931826e-02F, 2.380668715e-02F, 5.693469558e-03F, 6.114134862e-04F,
       2.948308997e-05F},
      {1.409871304e-02F, 2.224777669e-02F, 1.576426530e-02F, 5.015807104e-03F, 7.166182932e-04F, 4.597427900e-05F,
       1.324408431e-06F},
  };
  for (size_t i = 0; i < 7; ++i)
  {
    for (size_t j = 0; j < 7; ++j)
    {
      CHECK(kernel(i, j) == Catch::Approx(expected[i][j]).epsilon(1e-4));
    }
  }
}
