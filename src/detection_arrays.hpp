/**
 * @file detection_arrays.hpp
 * @brief Detection table columns shared by the CPU and GPU plane-processing paths.
 */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace breifast
{
/**
 * @brief Detection table columns, one entry per island found across every band of a plane — the C++ analog of
 *        breifast.py's `DetectionTable`.
 */
struct DetectionArrays
{
private:
  std::vector<long> peak_y_;
  std::vector<long> peak_x_;
  std::vector<long> island_size_;

  std::vector<std::string> stokes_label_;
  std::vector<std::string> band_label_;

  std::vector<float> flux_;

  std::vector<float> local_noise_;
  std::vector<float> local_noise_median_ratio_;

  std::vector<float> signal_to_noise_;
  std::vector<float> signal_to_noise_rms_;
  std::vector<float> signal_to_noise_extended_box_;
  std::vector<float> signal_to_noise_local_mad_;
  std::vector<float> signal_to_noise_global_mad_;

  std::vector<float> self_noise_;
  std::vector<float> excess_noise_factor_;
  std::vector<float> beam_chi2_;
  std::vector<float> boost_;

public:
  const std::vector<long> &peak_y() const
  {
    return peak_y_;
  }
  const std::vector<long> &peak_x() const
  {
    return peak_x_;
  }
  const std::vector<long> &island_size() const
  {
    return island_size_;
  }
  const std::vector<std::string> &stokes_label() const
  {
    return stokes_label_;
  }
  const std::vector<std::string> &band_label() const
  {
    return band_label_;
  }
  const std::vector<float> &flux() const
  {
    return flux_;
  }
  const std::vector<float> &local_noise() const
  {
    return local_noise_;
  }
  const std::vector<float> &local_noise_median_ratio() const
  {
    return local_noise_median_ratio_;
  }
  const std::vector<float> &signal_to_noise() const
  {
    return signal_to_noise_;
  }
  const std::vector<float> &signal_to_noise_rms() const
  {
    return signal_to_noise_rms_;
  }
  const std::vector<float> &signal_to_noise_extended_box() const
  {
    return signal_to_noise_extended_box_;
  }
  const std::vector<float> &signal_to_noise_local_mad() const
  {
    return signal_to_noise_local_mad_;
  }
  const std::vector<float> &signal_to_noise_global_mad() const
  {
    return signal_to_noise_global_mad_;
  }
  const std::vector<float> &self_noise() const
  {
    return self_noise_;
  }
  const std::vector<float> &excess_noise_factor() const
  {
    return excess_noise_factor_;
  }
  const std::vector<float> &beam_chi2() const
  {
    return beam_chi2_;
  }
  const std::vector<float> &boost() const
  {
    return boost_;
  }

  void reserve(const size_t size)
  {
    peak_y_.reserve(size);
    peak_x_.reserve(size);
    island_size_.reserve(size);
    stokes_label_.reserve(size);
    band_label_.reserve(size);
    flux_.reserve(size);
    local_noise_.reserve(size);
    local_noise_median_ratio_.reserve(size);
    signal_to_noise_.reserve(size);
    signal_to_noise_rms_.reserve(size);
    signal_to_noise_extended_box_.reserve(size);
    signal_to_noise_local_mad_.reserve(size);
    signal_to_noise_global_mad_.reserve(size);
    self_noise_.reserve(size);
    excess_noise_factor_.reserve(size);
    beam_chi2_.reserve(size);
    boost_.reserve(size);
  }

  size_t size() const
  {
    return peak_x_.size();
  }

  void add(const long peak_y, const long peak_x, const long island_size, const std::string &stokes_label,
           const std::string &band_label, const float flux, const float local_noise,
           const float local_noise_median_ratio, const float signal_to_noise, const float signal_to_noise_rms,
           const float signal_to_noise_extended_box, const float signal_to_noise_local_mad,
           const float signal_to_noise_global_mad, const float self_noise, const float excess_noise_factor,
           const float beam_chi2, const float boost)
  {
    peak_x_.push_back(peak_x);
    peak_y_.push_back(peak_y);
    island_size_.push_back(island_size);
    stokes_label_.push_back(stokes_label);
    band_label_.push_back(band_label);
    flux_.push_back(flux);
    local_noise_.push_back(local_noise);
    local_noise_median_ratio_.push_back(local_noise_median_ratio);
    signal_to_noise_.push_back(signal_to_noise);
    signal_to_noise_rms_.push_back(signal_to_noise_rms);
    signal_to_noise_extended_box_.push_back(signal_to_noise_extended_box);
    signal_to_noise_local_mad_.push_back(signal_to_noise_local_mad);
    signal_to_noise_global_mad_.push_back(signal_to_noise_global_mad);
    self_noise_.push_back(self_noise);
    excess_noise_factor_.push_back(excess_noise_factor);
    beam_chi2_.push_back(beam_chi2);
    boost_.push_back(boost);
  }
};
} // namespace breifast
