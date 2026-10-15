#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cpu/islands.hpp"

#include "../test_helpers.hpp"

using namespace breifast::islands;

TEST_CASE("label_islands on an all-false mask finds no islands", "[label_islands]")
{
  const auto mask = make_mask({{false, false, false}, {false, false, false}, {false, false, false}});
  const auto image = make_plane({{1.F, 2.F, 3.F}, {4.F, 5.F, 6.F}, {7.F, 8.F, 9.F}});
  const auto local_noise = make_plane({{1.F, 1.F, 1.F}, {1.F, 1.F, 1.F}, {1.F, 1.F, 1.F}});

  const auto result = label_islands(mask, image, local_noise);

  CHECK(result.peak_per_label.empty());
  for (size_t i_y = 0; i_y < 3; ++i_y)
  {
    for (size_t i_x = 0; i_x < 3; ++i_x)
    {
      CHECK(result.labels(i_y, i_x) == 0);
    }
  }
}

TEST_CASE("label_islands on a single rectangular island reports correct size, peak and local_noise",
         "[label_islands]")
{
  const auto mask = make_mask({{false, true, true}, {false, true, true}, {false, false, false}});
  const auto image = make_plane({{0.F, 2.F, -9.F}, {0.F, 5.F, 3.F}, {0.F, 0.F, 0.F}});
  const auto local_noise = make_plane({{0.F, 1.F, 4.F}, {0.F, 2.F, 3.F}, {0.F, 0.F, 0.F}});

  const auto result = label_islands(mask, image, local_noise);

  REQUIRE(result.peak_per_label.size() == 1);
  const auto &peak = result.peak_per_label.front();
  CHECK(peak.island_size == 4);
  CHECK(peak.value == Catch::Approx(9.F)); // |-9| is the largest abs value in the island
  CHECK(peak.coordinates.first == 0);      // y
  CHECK(peak.coordinates.second == 2);     // x
  CHECK(peak.local_noise == Catch::Approx(4.F));

  const auto label = result.labels(0, 2);
  CHECK(label != 0);
  for (const auto &[i_y, i_x] : std::vector<std::pair<size_t, size_t>>{{0, 1}, {0, 2}, {1, 1}, {1, 2}})
  {
    CHECK(result.labels(i_y, i_x) == label);
  }
  for (const auto &[i_y, i_x] : std::vector<std::pair<size_t, size_t>>{{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}})
  {
    CHECK(result.labels(i_y, i_x) == 0);
  }
}

TEST_CASE("label_islands assigns two disjoint islands independent, distinct labels", "[label_islands]")
{
  const auto mask = make_mask({{true, true, false, false, true}, {false, false, false, false, false}});
  const auto image = make_plane({{-3.F, 5.F, 0.F, 0.F, 7.F}, {0.F, 0.F, 0.F, 0.F, 0.F}});
  const auto local_noise = make_plane({{1.F, 2.F, 0.F, 0.F, 9.F}, {0.F, 0.F, 0.F, 0.F, 0.F}});

  const auto result = label_islands(mask, image, local_noise);

  REQUIRE(result.peak_per_label.size() == 2);
  const auto label_a = result.labels(0, 0);
  const auto label_b = result.labels(0, 4);
  CHECK(label_a != 0);
  CHECK(label_b != 0);
  CHECK(label_a != label_b);
  CHECK(result.labels(0, 1) == label_a);

  const auto &peak_a = result.peak_per_label.at(label_a - 1);
  CHECK(peak_a.island_size == 2);
  CHECK(peak_a.value == Catch::Approx(5.F));
  CHECK(peak_a.coordinates.first == 0);
  CHECK(peak_a.coordinates.second == 1);
  CHECK(peak_a.local_noise == Catch::Approx(2.F));

  const auto &peak_b = result.peak_per_label.at(label_b - 1);
  CHECK(peak_b.island_size == 1);
  CHECK(peak_b.value == Catch::Approx(7.F));
  CHECK(peak_b.local_noise == Catch::Approx(9.F));
}

TEST_CASE("label_islands does not merge diagonally-adjacent pixels (4-connectivity only)", "[label_islands]")
{
  // A checkerboard pattern: every true pixel only touches other true pixels diagonally, never
  // up/down/left/right, so every one must be its own island.
  const auto mask = make_mask({{true, false, true}, {false, true, false}, {true, false, true}});
  const auto image = make_plane({{1.F, 0.F, 1.F}, {0.F, 1.F, 0.F}, {1.F, 0.F, 1.F}});
  const auto local_noise = make_plane({{1.F, 0.F, 1.F}, {0.F, 1.F, 0.F}, {1.F, 0.F, 1.F}});

  const auto result = label_islands(mask, image, local_noise);

  CHECK(result.peak_per_label.size() == 5);
  for (const auto &peak : result.peak_per_label)
  {
    CHECK(peak.island_size == 1);
  }
}

TEST_CASE("label_islands merges a bridge shape into a single island via union-find", "[label_islands]")
{
  // Two provisional labels started on row 0 (columns 0 and 2) that only turn out to be the same
  // island once row 1's bridging pixel (column 1) is reached -- exercises unite(), not just simple
  // rectangles that never need a merge.
  const auto mask = make_mask({{true, false, true}, {true, true, true}});
  const auto image = make_plane({{2.F, 0.F, -8.F}, {1.F, 1.F, 1.F}});
  const auto local_noise = make_plane({{1.F, 0.F, 1.F}, {1.F, 1.F, 1.F}});

  const auto result = label_islands(mask, image, local_noise);

  REQUIRE(result.peak_per_label.size() == 1);
  const auto &peak = result.peak_per_label.front();
  CHECK(peak.island_size == 5);
  CHECK(peak.value == Catch::Approx(8.F));
  CHECK(peak.coordinates.first == 0);
  CHECK(peak.coordinates.second == 2);

  const auto label = result.labels(0, 0);
  CHECK(label != 0);
  CHECK(result.labels(0, 2) == label);
  CHECK(result.labels(1, 0) == label);
  CHECK(result.labels(1, 1) == label);
  CHECK(result.labels(1, 2) == label);
}

TEST_CASE("label_islands matches a seeded-random 5x6 mask against a Python reference", "[label_islands]")
{
  // Mask/image/local_noise and every expected island stat generated by an independent Python
  // implementation (flood fill in raster order, so label IDs are assigned in the same first-encounter
  // order as the C++ two-pass union-find implementation, numpy RNG seed=11).
  const auto mask = make_mask({{true, false, false, true, true, false},
                               {true, true, false, false, true, false},
                               {false, true, true, false, false, false},
                               {false, false, false, true, false, false},
                               {true, false, true, false, false, true}});
  const auto image = make_plane({{-0.660F, -4.460F, -8.340F, 7.920F, -1.400F, -7.050F},
                                 {3.470F, -5.960F, 8.030F, -5.660F, -9.340F, -5.980F},
                                 {-3.090F, -0.620F, 8.120F, 3.950F, -3.210F, -9.660F},
                                 {-6.800F, 9.930F, -0.810F, 3.820F, -8.910F, -9.320F},
                                 {6.920F, 1.760F, -3.830F, -3.650F, -8.220F, -6.550F}});
  const auto local_noise = make_plane({{0.220F, 4.210F, 2.380F, 0.720F, 3.720F, 1.060F},
                                       {0.400F, 3.030F, 4.490F, 0.230F, 4.050F, 1.030F},
                                       {0.560F, 0.190F, 1.540F, 3.660F, 2.520F, 4.280F},
                                       {1.160F, 1.640F, 1.360F, 4.890F, 4.710F, 1.770F},
                                       {2.240F, 1.640F, 3.760F, 0.300F, 0.430F, 2.080F}});

  const auto result = label_islands(mask, image, local_noise);

  const size_t expected_labels[5][6] = {
      {1, 0, 0, 2, 2, 0}, {1, 1, 0, 0, 2, 0}, {0, 1, 1, 0, 0, 0}, {0, 0, 0, 3, 0, 0}, {4, 0, 5, 0, 0, 6}};
  for (size_t i_y = 0; i_y < 5; ++i_y)
  {
    for (size_t i_x = 0; i_x < 6; ++i_x)
    {
      CHECK(result.labels(i_y, i_x) == expected_labels[i_y][i_x]);
    }
  }

  REQUIRE(result.peak_per_label.size() == 6);
  struct Expected
  {
    size_t label, size, y, x;
    float value, local_noise;
  };
  const std::vector<Expected> expected = {
      {1, 5, 2, 2, 8.120F, 3.030F}, {2, 3, 1, 4, 9.340F, 4.050F}, {3, 1, 3, 3, 3.820F, 4.890F},
      {4, 1, 4, 0, 6.920F, 2.240F}, {5, 1, 4, 2, 3.830F, 3.760F}, {6, 1, 4, 5, 6.550F, 2.080F},
  };
  for (const auto &e : expected)
  {
    const auto &peak = result.peak_per_label.at(e.label - 1);
    CHECK(peak.island_size == e.size);
    CHECK(peak.coordinates.first == e.y);
    CHECK(peak.coordinates.second == e.x);
    CHECK(peak.value == Catch::Approx(e.value).epsilon(1e-4));
    CHECK(peak.local_noise == Catch::Approx(e.local_noise).epsilon(1e-4));
  }
}

TEST_CASE("label_islands matches a seeded-random 8x8 mask against a Python reference", "[label_islands]")
{
  // Same cross-check as the 5x6 case above, larger and denser (numpy RNG seed=22), including a
  // multi-pixel island (label 4, size 4) and an island that touches the array's bottom-right border.
  const auto mask = make_mask({{false, true, true, false, false, false, false, false},
                               {true, false, false, true, false, true, true, false},
                               {false, true, false, false, false, true, true, false},
                               {false, false, false, false, false, false, false, false},
                               {true, false, false, false, false, true, true, false},
                               {false, true, false, false, false, false, false, false},
                               {true, false, false, false, false, false, false, false},
                               {false, false, false, false, false, true, true, true}});
  const auto image = make_plane({{1.840F, 3.190F, 6.390F, -4.260F, 0.320F, -0.930F, -6.590F, 9.630F},
                                 {-3.290F, 3.810F, 3.830F, 5.420F, 6.670F, 8.930F, -3.580F, 3.370F},
                                 {-3.660F, -4.040F, 5.860F, 4.990F, 4.640F, -7.540F, 9.340F, 2.570F},
                                 {-0.700F, 8.260F, -7.770F, -3.970F, -7.140F, -3.450F, 5.740F, -7.000F},
                                 {5.230F, 1.960F, -3.410F, 7.000F, -9.150F, -5.130F, 0.000F, -2.000F},
                                 {-2.300F, 9.940F, 0.980F, 3.220F, -0.880F, -6.030F, -7.920F, 5.110F},
                                 {4.460F, 9.830F, -4.970F, -2.120F, -2.490F, 1.470F, -9.040F, 5.950F},
                                 {4.660F, -8.120F, 4.250F, -3.210F, -1.360F, -5.970F, 3.760F, 1.560F}});
  const auto local_noise = make_plane({{0.210F, 0.830F, 3.690F, 0.760F, 2.390F, 3.380F, 0.930F, 2.540F},
                                       {2.160F, 0.790F, 3.960F, 0.500F, 4.150F, 1.700F, 1.860F, 1.200F},
                                       {4.290F, 4.990F, 1.020F, 3.180F, 0.310F, 2.050F, 1.180F, 3.200F},
                                       {1.090F, 3.100F, 4.810F, 1.330F, 0.730F, 4.300F, 0.360F, 3.730F},
                                       {4.470F, 0.760F, 2.620F, 4.880F, 0.840F, 4.190F, 3.390F, 0.450F},
                                       {0.940F, 4.630F, 0.420F, 1.840F, 0.270F, 1.360F, 1.150F, 3.970F},
                                       {0.500F, 0.650F, 2.760F, 1.080F, 2.650F, 0.360F, 1.020F, 1.780F},
                                       {3.620F, 4.160F, 1.030F, 2.580F, 4.290F, 2.710F, 3.980F, 4.270F}});

  const auto result = label_islands(mask, image, local_noise);

  const size_t expected_labels[8][8] = {
      {0, 1, 1, 0, 0, 0, 0, 0}, {2, 0, 0, 3, 0, 4, 4, 0}, {0, 5, 0, 0, 0, 4, 4, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
      {6, 0, 0, 0, 0, 7, 7, 0}, {0, 8, 0, 0, 0, 0, 0, 0}, {9, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 10, 10, 10}};
  for (size_t i_y = 0; i_y < 8; ++i_y)
  {
    for (size_t i_x = 0; i_x < 8; ++i_x)
    {
      CHECK(result.labels(i_y, i_x) == expected_labels[i_y][i_x]);
    }
  }

  REQUIRE(result.peak_per_label.size() == 10);
  struct Expected
  {
    size_t label, size, y, x;
    float value, local_noise;
  };
  const std::vector<Expected> expected = {
      {1, 2, 0, 2, 6.390F, 3.690F}, {2, 1, 1, 0, 3.290F, 2.160F}, {3, 1, 1, 3, 5.420F, 0.500F},
      {4, 4, 2, 6, 9.340F, 2.050F}, {5, 1, 2, 1, 4.040F, 4.990F}, {6, 1, 4, 0, 5.230F, 4.470F},
      {7, 2, 4, 5, 5.130F, 4.190F}, {8, 1, 5, 1, 9.940F, 4.630F}, {9, 1, 6, 0, 4.460F, 0.500F},
      {10, 3, 7, 5, 5.970F, 4.270F},
  };
  for (const auto &e : expected)
  {
    const auto &peak = result.peak_per_label.at(e.label - 1);
    CHECK(peak.island_size == e.size);
    CHECK(peak.coordinates.first == e.y);
    CHECK(peak.coordinates.second == e.x);
    CHECK(peak.value == Catch::Approx(e.value).epsilon(1e-4));
    CHECK(peak.local_noise == Catch::Approx(e.local_noise).epsilon(1e-4));
  }
}
