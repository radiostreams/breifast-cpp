#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "host_copies.hpp"

namespace
{
std::vector<std::uint8_t> pattern(const std::size_t size, const std::uint8_t seed)
{
  std::vector<std::uint8_t> values(size);
  for (std::size_t i = 0; i < size; ++i)
  {
    values[i] = static_cast<std::uint8_t>(seed + i * 7 + i / 251);
  }
  return values;
}
} // namespace

TEST_CASE("HostCopies copies every registered array, whatever its size", "[host_copies]")
{
  // Empty, smaller than a chunk, exactly one chunk, a chunk and a bit, several chunks. The chunk is 16 bytes.
  const std::vector<std::size_t> sizes = {0, 1, 15, 16, 17, 100, 4096};
  std::vector<std::vector<std::uint8_t>> sources;
  std::vector<std::vector<std::uint8_t>> destinations;
  breifast::HostCopies copies;
  for (std::size_t i = 0; i < sizes.size(); ++i)
  {
    sources.push_back(pattern(sizes[i], static_cast<std::uint8_t>(i * 31)));
    destinations.emplace_back(sizes[i], std::uint8_t{0xEE});
    copies.add(sources[i].data(), destinations[i].data(), sizes[i]);
  }

  copies.run(16);

  for (std::size_t i = 0; i < sizes.size(); ++i)
  {
    CHECK(destinations[i] == sources[i]);
  }
}

TEST_CASE("HostCopies with nothing registered does nothing", "[host_copies]")
{
  const breifast::HostCopies copies;
  copies.run();
}

TEST_CASE("HostCopies copies a large array with the default chunk size", "[host_copies]")
{
  const std::vector<std::uint8_t> source = pattern((std::size_t{9} << 20) + 123, 5);
  std::vector<std::uint8_t> destination(source.size(), std::uint8_t{0});
  breifast::HostCopies copies;
  copies.add(source.data(), destination.data(), source.size());

  copies.run();

  CHECK(destination == source);
}
