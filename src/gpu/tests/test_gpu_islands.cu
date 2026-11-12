#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "gpu_band.hpp"
#include "gpu_kernels.hpp"
#include "gpu_test_utils.hpp"

using breifast::gpu::DeviceVector;
using breifast::gpu::gpu_label_islands;
using breifast::gpu::LabelScratch;
using breifast::gpu::Stream;
using breifast::gpu::test::download;
using breifast::gpu::test::download_scalar;
using breifast::gpu::test::upload;
using Catch::Matchers::Equals;

namespace
{
std::vector<unsigned int> run_label(const std::vector<unsigned char> &mask, size_t n_y, size_t n_x,
                                    unsigned int *out_count = nullptr)
{
  const size_t n = mask.size();

  const auto device_mask = upload(std::vector<bool>(mask.begin(), mask.end()));
  DeviceVector<unsigned int> device_labels(n);
  DeviceVector<unsigned int> device_count(1);
  LabelScratch scratch;
  scratch.ensure(n);

  const Stream stream;
  gpu_label_islands(device_mask, n_y, n_x, device_labels, device_count, scratch, stream.get());
  const auto host_labels = download(device_labels, stream);
  const unsigned int count = download_scalar(device_count.data(), stream);

  if (out_count != nullptr)
  {
    *out_count = count;
  }
  return std::vector<unsigned int>(host_labels.begin(), host_labels.end());
}

std::vector<unsigned int> host_reference(const std::vector<unsigned char> &mask, size_t n_y, size_t n_x)
{
  const size_t n = n_y * n_x;
  std::vector<unsigned int> labels(n, 0);
  std::vector<size_t> stack;
  unsigned int next_label = 0;

  for (size_t start = 0; start < n; ++start)
  {
    if (mask[start] == 0 || labels[start] != 0)
    {
      continue;
    }

    ++next_label;
    labels[start] = next_label;
    stack.push_back(start);
    while (!stack.empty())
    {
      const size_t index = stack.back();
      stack.pop_back();
      const size_t i_y = index / n_x;
      const size_t i_x = index % n_x;

      const auto visit = [&](size_t neighbour)
      {
        if (mask[neighbour] != 0 && labels[neighbour] == 0)
        {
          labels[neighbour] = next_label;
          stack.push_back(neighbour);
        }
      };
      if (i_x > 0)
      {
        visit(index - 1);
      }
      if (i_x + 1 < n_x)
      {
        visit(index + 1);
      }
      if (i_y > 0)
      {
        visit(index - n_x);
      }
      if (i_y + 1 < n_y)
      {
        visit(index + n_x);
      }
    }
  }
  return labels;
}
} // namespace

TEST_CASE("gpu_label_islands numbers 4-connected islands in raster order of their first pixel", "[gpu][islands]")
{
  CHECK_THAT(run_label({1, 0, 1, 1, 1, 0, 0, 0, 1}, 3, 3), Equals(std::vector<unsigned int>{1, 0, 2, 1, 1, 0, 0, 0, 3}));
  CHECK_THAT(run_label({1, 0, 0, 1}, 1, 4), Equals(std::vector<unsigned int>{1, 0, 0, 2}));
  CHECK_THAT(run_label({0, 1, 0, 1, 1, 1, 0, 1, 0}, 3, 3),
             Equals(std::vector<unsigned int>{0, 1, 0, 1, 1, 1, 0, 1, 0}));
  CHECK_THAT(run_label({1, 1, 1, 1}, 2, 2), Equals(std::vector<unsigned int>{1, 1, 1, 1}));
}

TEST_CASE("gpu_label_islands keeps corner-touching cells separate under 4-connectivity", "[gpu][islands]")
{
  CHECK_THAT(run_label({1, 0, 0, 1}, 2, 2), Equals(std::vector<unsigned int>{1, 0, 0, 2}));
  CHECK_THAT(run_label({0, 1, 1, 0}, 2, 2), Equals(std::vector<unsigned int>{0, 1, 2, 0}));
  CHECK_THAT(run_label({1, 0, 1, 1}, 4, 1), Equals(std::vector<unsigned int>{1, 0, 2, 2}));
}

TEST_CASE("gpu_label_islands matches a host reference on a seeded multi-block mask", "[gpu][islands]")
{
  const size_t n_y = 37;
  const size_t n_x = 53;
  std::vector<unsigned char> mask(n_y * n_x);
  for (size_t i = 0; i < mask.size(); ++i)
  {
    mask[i] = static_cast<unsigned char>((i * 11 + 5) % 7 < 3);
  }

  unsigned int count = 0;
  CHECK_THAT(run_label(mask, n_y, n_x, &count), Equals(host_reference(mask, n_y, n_x)));
  CHECK(count > 1U);
}

TEST_CASE("gpu_label_islands propagates along a long thin island", "[gpu][islands]")
{
  const size_t n_x = 300;
  const std::vector<unsigned char> mask(n_x, 1);

  unsigned int count = 0;
  CHECK_THAT(run_label(mask, 1, n_x, &count), Equals(std::vector<unsigned int>(n_x, 1)));
  CHECK(count == 1U);
}

TEST_CASE("gpu_label_islands converges on a large multi-block mask", "[gpu][islands]")
{
  const size_t n_y = 256;
  const size_t n_x = 256;
  std::vector<unsigned char> mask(n_y * n_x, 0);
  for (size_t x = 0; x < n_x; ++x)
  {
    mask[x] = 1;
  }
  for (size_t y = 1; y < n_y; ++y)
  {
    for (size_t x = 0; x < n_x; x += 16)
    {
      mask[y * n_x + x] = 1;
    }
  }

  unsigned int count = 0;
  CHECK_THAT(run_label(mask, n_y, n_x, &count), Equals(host_reference(mask, n_y, n_x)));
  CHECK(count == 1U);
}

TEST_CASE("gpu_label_islands handles all-false and all-true masks", "[gpu][islands]")
{
  unsigned int count = 0;
  CHECK_THAT(run_label(std::vector<unsigned char>(6, 0), 2, 3, &count), Equals(std::vector<unsigned int>(6, 0)));
  CHECK(count == 0U);

  CHECK_THAT(run_label(std::vector<unsigned char>(6, 1), 2, 3, &count), Equals(std::vector<unsigned int>(6, 1)));
  CHECK(count == 1U);
}
