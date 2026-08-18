#pragma once
#include <initializer_list>

#include <xtensor/containers/xarray.hpp>

inline xt::xarray<float> make_plane(std::initializer_list<std::initializer_list<float>> rows)
{
  const size_t n_y = rows.size();
  const size_t n_x = rows.begin()->size();
  xt::xarray<float> plane;
  plane.resize({n_y, n_x});
  size_t i_y = 0;
  for (const auto &row : rows)
  {
    size_t i_x = 0;
    for (const auto value : row)
    {
      plane(i_y, i_x) = value;
      ++i_x;
    }
    ++i_y;
  }
  return plane;
}

inline xt::xarray<float> make_cube(std::initializer_list<std::initializer_list<std::initializer_list<float>>> channels)
{
  const size_t n_channels = channels.size();
  const size_t n_y = channels.begin()->size();
  const size_t n_x = channels.begin()->begin()->size();
  xt::xarray<float> cube;
  cube.resize({n_channels, n_y, n_x});
  size_t i_channel = 0;
  for (const auto &rows : channels)
  {
    size_t i_y = 0;
    for (const auto &row : rows)
    {
      size_t i_x = 0;
      for (const auto value : row)
      {
        cube(i_channel, i_y, i_x) = value;
        ++i_x;
      }
      ++i_y;
    }
    ++i_channel;
  }
  return cube;
}

inline xt::xarray<bool> make_mask(std::initializer_list<std::initializer_list<bool>> rows)
{
  const size_t n_y = rows.size();
  const size_t n_x = rows.begin()->size();
  xt::xarray<bool> mask;
  mask.resize({n_y, n_x});
  size_t i_y = 0;
  for (const auto &row : rows)
  {
    size_t i_x = 0;
    for (const auto value : row)
    {
      mask(i_y, i_x) = value;
      ++i_x;
    }
    ++i_y;
  }
  return mask;
}

inline xt::xarray<bool> make_mask1d(std::initializer_list<bool> values)
{
  xt::xarray<bool> mask;
  mask.resize({values.size()});
  size_t i = 0;
  for (const auto value : values)
  {
    mask(i) = value;
    ++i;
  }
  return mask;
}
