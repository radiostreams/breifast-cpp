#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpu/island_stamps.hpp"

using breifast::plane_lot::detail::stamp_extract;

namespace
{
constexpr size_t n_y = 6;
constexpr size_t n_x = 7;

// plane(y, x) = 100 + 10 * y + x, so every pixel is nonzero and the zero fill is easy to see.
const std::vector<float> plane = {
    100.F, 101.F, 102.F, 103.F, 104.F, 105.F, 106.F,
    110.F, 111.F, 112.F, 113.F, 114.F, 115.F, 116.F,
    120.F, 121.F, 122.F, 123.F, 124.F, 125.F, 126.F,
    130.F, 131.F, 132.F, 133.F, 134.F, 135.F, 136.F,
    140.F, 141.F, 142.F, 143.F, 144.F, 145.F, 146.F,
    150.F, 151.F, 152.F, 153.F, 154.F, 155.F, 156.F,
};

std::vector<float> run_stamp_extract(long centre_y, long centre_x, long radius, float prefill = 0.F)
{
  const size_t size = static_cast<size_t>(2 * radius + 1);
  std::vector<float> stamp(size * size, prefill);
  stamp_extract(plane, n_y, n_x, centre_y, centre_x, radius, stamp);
  return stamp;
}
} // namespace

TEST_CASE("stamp_extract matches numpy's padded slicing for windows inside, on the edges and at the corners",
          "[stamp_extract]")
{
  // numpy: np.pad(plane, 2)[cy:cy + 5, cx:cx + 5], the 5x5 stamp around (cy, cx). The window of (3, 3) is inside the
  // plane, the others stick out on one or two sides.
  struct Case
  {
    long centre_y;
    long centre_x;
    std::vector<float> expected;
  };
  const std::vector<Case> cases = {
      {3, 3, {
        111.F, 112.F, 113.F, 114.F, 115.F,
        121.F, 122.F, 123.F, 124.F, 125.F,
        131.F, 132.F, 133.F, 134.F, 135.F,
        141.F, 142.F, 143.F, 144.F, 145.F,
        151.F, 152.F, 153.F, 154.F, 155.F,
      }},
      {0, 0, {
        0.F, 0.F, 0.F, 0.F, 0.F,
        0.F, 0.F, 0.F, 0.F, 0.F,
        0.F, 0.F, 100.F, 101.F, 102.F,
        0.F, 0.F, 110.F, 111.F, 112.F,
        0.F, 0.F, 120.F, 121.F, 122.F,
      }},
      {5, 6, {
        134.F, 135.F, 136.F, 0.F, 0.F,
        144.F, 145.F, 146.F, 0.F, 0.F,
        154.F, 155.F, 156.F, 0.F, 0.F,
        0.F, 0.F, 0.F, 0.F, 0.F,
        0.F, 0.F, 0.F, 0.F, 0.F,
      }},
      {0, 3, {
        0.F, 0.F, 0.F, 0.F, 0.F,
        0.F, 0.F, 0.F, 0.F, 0.F,
        101.F, 102.F, 103.F, 104.F, 105.F,
        111.F, 112.F, 113.F, 114.F, 115.F,
        121.F, 122.F, 123.F, 124.F, 125.F,
      }},
      {3, 6, {
        114.F, 115.F, 116.F, 0.F, 0.F,
        124.F, 125.F, 126.F, 0.F, 0.F,
        134.F, 135.F, 136.F, 0.F, 0.F,
        144.F, 145.F, 146.F, 0.F, 0.F,
        154.F, 155.F, 156.F, 0.F, 0.F,
      }},
  };

  for (const auto &c : cases)
  {
    INFO("centre (" << c.centre_y << ", " << c.centre_x << ")");
    CHECK(run_stamp_extract(c.centre_y, c.centre_x, 2) == c.expected);
  }
}

TEST_CASE("stamp_extract pads a window larger than the plane with zeros on every side", "[stamp_extract]")
{
  // numpy: np.pad(plane, 8)[2:19, 3:20], the 17x17 stamp around (2, 3): the whole plane sits inside a frame of zeros.
  const std::vector<float> expected = {
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 100.F, 101.F, 102.F, 103.F, 104.F, 105.F, 106.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 110.F, 111.F, 112.F, 113.F, 114.F, 115.F, 116.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 120.F, 121.F, 122.F, 123.F, 124.F, 125.F, 126.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 130.F, 131.F, 132.F, 133.F, 134.F, 135.F, 136.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 140.F, 141.F, 142.F, 143.F, 144.F, 145.F, 146.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 150.F, 151.F, 152.F, 153.F, 154.F, 155.F, 156.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
  };

  CHECK(run_stamp_extract(2, 3, 8) == expected);
}

TEST_CASE("stamp_extract returns zeros when the window does not touch the plane", "[stamp_extract]")
{
  const std::vector<float> zeros(25, 0.F);

  // Far above, far to the left, far below and far to the right, and just one pixel beyond each edge.
  for (const auto &[centre_y, centre_x] : {std::pair<long, long>{-10, 3}, {3, -10}, {100, 3}, {3, 100}, {-3, 3}, {3, -3},
                                          {8, 3}, {3, 10}})
  {
    INFO("centre (" << centre_y << ", " << centre_x << ")");
    CHECK(run_stamp_extract(centre_y, centre_x, 2) == zeros);
  }
}

TEST_CASE("stamp_extract with radius 0 is the pixel itself or 0 outside the plane", "[stamp_extract]")
{
  CHECK(run_stamp_extract(0, 0, 0) == std::vector<float>{100.F});
  CHECK(run_stamp_extract(5, 6, 0) == std::vector<float>{156.F});
  CHECK(run_stamp_extract(2, 4, 0) == std::vector<float>{124.F});
  CHECK(run_stamp_extract(6, 0, 0) == std::vector<float>{0.F});
  CHECK(run_stamp_extract(0, -1, 0) == std::vector<float>{0.F});
}

TEST_CASE("stamp_extract overwrites the stale values of out_stamp", "[stamp_extract]")
{
  // out_stamp is per-thread scratch holding the previous island: the pixels outside the plane must read 0, not whatever
  // was there. The corner window has 16 of its 25 pixels outside the plane.
  const auto from_zero = run_stamp_extract(0, 0, 2);

  CHECK(run_stamp_extract(0, 0, 2, std::numeric_limits<float>::quiet_NaN()) == from_zero);
  CHECK(run_stamp_extract(0, 0, 2, 1.0e30F) == from_zero);
  CHECK(run_stamp_extract(-10, 3, 2, 1.0e30F) == std::vector<float>(25, 0.F));
}

TEST_CASE("stamp_extract rejects a plane or stamp of the wrong size and a negative radius", "[stamp_extract]")
{
  std::vector<float> stamp(25);
  std::vector<float> short_stamp(24);
  const std::vector<float> short_plane(n_y * n_x - 1);

  using Catch::Matchers::ContainsSubstring;
  CHECK_THROWS_WITH(stamp_extract(short_plane, n_y, n_x, 3, 3, 2, stamp), ContainsSubstring("plane size mismatch"));
  CHECK_THROWS_WITH(stamp_extract(plane, n_y, n_x, 3, 3, 2, short_stamp), ContainsSubstring("out_stamp size mismatch"));
  CHECK_THROWS_WITH(stamp_extract(plane, n_y, n_x, 3, 3, -1, stamp), ContainsSubstring("negative radius"));
  CHECK_NOTHROW(stamp_extract(plane, n_y, n_x, 3, 3, 2, stamp));
}
