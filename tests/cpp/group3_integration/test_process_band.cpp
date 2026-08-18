#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "plane_lot_dispatch.hpp"

#include "../test_helpers.hpp"

using namespace breifast;

TEST_CASE("process_band detects a single clear point-source island with correct identity and basic detection "
         "fields",
         "[process_band]")
{
  // 7x7 plane, single-channel band: background = 1.0 everywhere except the center pixel (3,3) = 50.0. The peak is a
  // 1-in-49 outlier, so it can never move the median -- keeping noise_global_mad trivially hand-computable.
  auto plane_rows = std::vector<std::vector<float>>(7, std::vector<float>(7, 1.F));
  plane_rows[3][3] = 50.F;
  xt::xarray<float> plane_2d;
  plane_2d.resize({7, 7});
  for (size_t i_y = 0; i_y < 7; ++i_y)
  {
    for (size_t i_x = 0; i_x < 7; ++i_x)
    {
      plane_2d(i_y, i_x) = plane_rows[i_y][i_x];
    }
  }

  PlaneBatchItem item;
  item.stokes_cube.resize({1, 7, 7});
  xt::view(item.stokes_cube, 0) = plane_2d;
  item.weights = xt::xarray<float>({1.F});
  item.nonzero_channels = xt::xarray<bool>({true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({1, 7, 7});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({7, 7});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({7, 7});
  group.selfnoise_raw.fill(2.5F);
  group.dde_noise.resize({7, 7});
  group.dde_noise.fill(0.F);

  BandSlice band{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "bandA"};

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false; // Global-MAD noise -> a single broadcast constant, trivial to hand-verify;
                                          // the RMS-box estimator is already covered directly by
                                          // compute_rms_noise_map's own tests.
  config.boxsize = 3;
  config.inner_boxsize = 0; // Zeroes only the postage stamp's own center pixel, leaving its 8-neighbor donut intact.
  config.reject_noise_threshold = 0.F;
  config.threshold = 5.F; // Background snr ~0.67 fails this; the center's snr ~33.7 clears it -- exactly one
                          // size-1 island.
  config.beam_size_enabled = false;

  ProcessPlaneResult result;
  process_band(item, group, band, config, result);

  REQUIRE(result.detections.peak_y.size() == 1);
  CHECK(result.detections.peak_y[0] == 3);
  CHECK(result.detections.peak_x[0] == 3);
  CHECK(result.detections.island_size[0] == 1);
  CHECK(result.detections.stokes_label[0] == "I");
  CHECK(result.detections.band_label[0] == "bandA");

  // noise_global_mad = median(|plane|) * 1.4826 = 1.0 * 1.4826, since only 1 of 49 pixels differs from 1.0.
  const float expected_noise = 1.4826F;
  CHECK(result.detections.local_noise[0] == Catch::Approx(expected_noise).epsilon(1e-4));
  CHECK(result.detections.local_noise_median_ratio[0] == Catch::Approx(1.F).epsilon(1e-4));

  const float expected_snr = 50.F / expected_noise;
  CHECK(result.detections.signal_to_noise[0] == Catch::Approx(expected_snr).epsilon(1e-4));
  // Uniform background -> the postage stamp's 8-neighbor donut is all identical values, so its stddev is exactly 0
  // and signal_to_noise_rms hits the rms==0 guard.
  CHECK(result.detections.signal_to_noise_rms[0] == 0.F);
  CHECK(result.detections.signal_to_noise_local_mad[0] == Catch::Approx(expected_snr).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_global_mad[0] == Catch::Approx(expected_snr).epsilon(1e-4));

  // signal_to_noise_extended_box = flux * max_to_std_ratio_value(): computed via the same config call the
  // production code uses, rather than hardcoding a literal -- that ratio is itself the output of a numerical
  // search Group 1's max_to_std_ratio tests already pin down independently.
  const float expected_extended_box = 50.F * config.max_to_std_ratio_value();
  CHECK(result.detections.signal_to_noise_extended_box[0] == Catch::Approx(expected_extended_box).epsilon(1e-4));

  CHECK(result.detections.self_noise[0] == Catch::Approx(2.5F));
  CHECK(result.detections.excess_noise_factor[0] == Catch::Approx(1.F).epsilon(1e-4));
  CHECK(result.detections.beam_chi2[0] == 0.F);
  CHECK(result.detections.boost[0] == 0.F);
}

TEST_CASE("process_band zero-pads postage stamps correctly for islands touching all four plane edges",
         "[process_band]")
{
  // 5x5 plane, boxsize=3 (radius=1, pad=2). Top-left peak (0,0)=50 pulls its postage stamp's top/left edges into
  // padded index 1 (< pad) -- the duplicate-condition bug fixed earlier. Bottom-right peak (4,4)=60 pulls its
  // stamp's bottom/right edges into padded index 7 (== pad+n_y == pad+n_x) -- the off-by-one bug fixed just now.
  // Both islands share the same decoy neighbor set {2,4,6} by design, so their noise-derived stats are directly
  // comparable and any padding regression (wrapped/garbage data instead of zeros) would corrupt both differently.
  xt::xarray<float> plane_2d;
  plane_2d.resize({5, 5});
  plane_2d.fill(1.F);
  plane_2d(0, 0) = 50.F; // top-left peak
  plane_2d(0, 1) = 2.F;
  plane_2d(1, 0) = 4.F;
  plane_2d(1, 1) = 6.F;
  plane_2d(4, 4) = 60.F; // bottom-right peak
  plane_2d(4, 3) = 6.F;
  plane_2d(3, 4) = 4.F;
  plane_2d(3, 3) = 2.F;

  PlaneBatchItem item;
  item.stokes_cube.resize({1, 5, 5});
  xt::view(item.stokes_cube, 0) = plane_2d;
  item.weights = xt::xarray<float>({1.F});
  item.nonzero_channels = xt::xarray<bool>({true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({1, 5, 5});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({5, 5});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({5, 5});
  group.selfnoise_raw.fill(2.5F);
  group.selfnoise_raw(0, 0) = 1.1F; // distinct per-peak values, to prove each detection reads its own pixel.
  group.selfnoise_raw(4, 4) = 9.9F;
  group.dde_noise.resize({5, 5});
  group.dde_noise.fill(0.F);

  BandSlice band{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "bandA"};

  PlaneConfig config;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  // Peaks (snr ~33.7, ~40.5) clear this; every decoy/background value (max snr ~4.05) doesn't -- exactly two
  // size-1 islands.
  config.threshold = 5.F;
  config.beam_size_enabled = false;

  ProcessPlaneResult result;
  process_band(item, group, band, config, result);

  REQUIRE(result.detections.peak_y.size() == 2);
  // Detection order isn't guaranteed (islands come from an unordered_map), so look each one up by its peak
  // coordinates rather than assuming index 0/1.
  const auto find_index = [&](long peak_y, long peak_x)
  {
    for (size_t i = 0; i < result.detections.peak_y.size(); ++i)
    {
      if (result.detections.peak_y[i] == peak_y && result.detections.peak_x[i] == peak_x)
      {
        return i;
      }
    }
    FAIL("no detection at (" << peak_y << ", " << peak_x << ")");
    return static_cast<size_t>(0);
  };
  const auto tl = find_index(0, 0);
  const auto br = find_index(4, 4);

  const float expected_noise = 1.4826F;
  const float expected_stddev = 1.632993F; // stddev of the shared decoy donut {2, 4, 6}.
  const float expected_lmad = 5.9304F;     // median({2,4,6}) * 1.4826.
  const auto ratio = config.max_to_std_ratio_value();

  CHECK(result.detections.island_size[tl] == 1);
  CHECK(result.detections.local_noise[tl] == Catch::Approx(expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise[tl] == Catch::Approx(50.F / expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_rms[tl] == Catch::Approx(50.F / expected_stddev).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_local_mad[tl] == Catch::Approx(50.F / expected_lmad).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_global_mad[tl] == Catch::Approx(50.F / expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_extended_box[tl] == Catch::Approx(50.F * ratio / 6.F).epsilon(1e-4));
  CHECK(result.detections.self_noise[tl] == Catch::Approx(1.1F));

  CHECK(result.detections.island_size[br] == 1);
  CHECK(result.detections.local_noise[br] == Catch::Approx(expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise[br] == Catch::Approx(60.F / expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_rms[br] == Catch::Approx(60.F / expected_stddev).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_local_mad[br] == Catch::Approx(60.F / expected_lmad).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_global_mad[br] == Catch::Approx(60.F / expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_extended_box[br] == Catch::Approx(60.F * ratio / 6.F).epsilon(1e-4));
  CHECK(result.detections.self_noise[br] == Catch::Approx(9.9F));
}

TEST_CASE("process_band computes a weighted, unequally-weighted multi-channel mean, with subtract_mean off and on",
         "[process_band]")
{
  // 5x5 plane, 2-channel band, weights=[3,1] (deliberately unequal -- a naive unweighted mean would give 40 at the
  // peak instead of the correct weighted 50, so this directly catches that bug class). Peak away from any edge, so
  // its postage stamp never touches padding (already covered by the border test above).
  xt::xarray<float> stokes_cube;
  stokes_cube.resize({2, 5, 5});
  xt::view(stokes_cube, 0) = 1.F;
  xt::view(stokes_cube, 1) = 1.F;
  stokes_cube(0, 2, 2) = 60.F;
  stokes_cube(1, 2, 2) = 20.F; // weighted mean at the peak: (3*60 + 1*20) / 4 = 50.

  PlaneBatchItem item;
  item.stokes_cube = stokes_cube;
  item.weights = xt::xarray<float>({3.F, 1.F});
  item.nonzero_channels = xt::xarray<bool>({true, true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({2, 5, 5});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({5, 5});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({5, 5});
  group.selfnoise_raw.fill(2.5F);
  group.dde_noise.resize({5, 5});
  group.dde_noise.fill(0.F);

  BandSlice band{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "bandA"};

  PlaneConfig config;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 5.F;
  config.beam_size_enabled = false;

  const auto ratio = config.max_to_std_ratio_value();
  const float expected_noise = 1.4826F; // median(|plane|) is unmoved by a 1-in-25 outlier, in both parts below.

  // Part A: subtract_mean off.
  {
    config.subtract_mean = false;
    ProcessPlaneResult result;
    process_band(item, group, band, config, result);

    REQUIRE(result.detections.peak_y.size() == 1);
    CHECK(result.detections.peak_y[0] == 2);
    CHECK(result.detections.peak_x[0] == 2);
    CHECK(result.detections.signal_to_noise[0] == Catch::Approx(50.F / expected_noise).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_global_mad[0] == Catch::Approx(50.F / expected_noise).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_rms[0] == 0.F); // uniform-1.0 donut, same as the first test's case.
    CHECK(result.detections.signal_to_noise_extended_box[0] == Catch::Approx(50.F * ratio).epsilon(1e-4));
  }

  // Part B: subtract_mean on. mean_image = 4.0 on channel 0, 0.0 on channel 1, uniform over the grid,
  // inv_band_size=0.5 -> subtracts exactly 0.5*(4+0)=2.0 at every pixel: background 1.0 -> -1.0, peak 50 -> 48. A
  // clean, exact delta from part A isolates the subtraction's effect from everything else (noise stays 1.4826,
  // since abs() doesn't care about the sign flip on the background).
  {
    xt::xarray<float> mean_image;
    mean_image.resize({2, 5, 5});
    xt::view(mean_image, 0) = 4.F;
    xt::view(mean_image, 1) = 0.F;
    group.mean_image = mean_image;
    config.subtract_mean = true;
    ProcessPlaneResult result;
    process_band(item, group, band, config, result);

    REQUIRE(result.detections.peak_y.size() == 1);
    CHECK(result.detections.peak_y[0] == 2);
    CHECK(result.detections.peak_x[0] == 2);
    CHECK(result.detections.signal_to_noise[0] == Catch::Approx(48.F / expected_noise).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_global_mad[0] == Catch::Approx(48.F / expected_noise).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_rms[0] == 0.F); // uniform-(-1.0) donut: still identical, still 0.
    CHECK(result.detections.signal_to_noise_extended_box[0] == Catch::Approx(48.F * ratio).epsilon(1e-4));
  }
}

TEST_CASE("process_band matches an independent Python reference on a wide-scale plane with the real RMS noise "
         "estimator",
         "[process_band]")
{
  // 12x12 plane, 1 channel, numpy seed=33, background uniform(-10,10), overwritten with 4 well-separated interior
  // point sources of alternating sign: (2,2)=70, (2,9)=-65, (9,2)=55, (9,9)=-80. boxsize=5/inner_boxsize=1/
  // threshold=4.0 keeps the background's max |snr| (~1.88) comfortably below threshold, giving exactly 4 clean
  // size-1 islands -- confirmed by an independent Python reference script (make_noise_map's box-RMS formula,
  // noisemap/excess_noise_factor combination, and postage-stamp extreme/rms/lmad, all re-derived from breifast.py's
  // algorithm, not from this C++ code). First Group 3 test to exercise the real windowed-RMS noise estimator
  // in-line (compute_rms_noise_map was previously only tested standalone in Group 1) and the first with non-1.0
  // excess_noise_factor / non-trivial selfnoise_scale_factor+ddenoise_scaling_factor combination.
  constexpr size_t n = 12;
  const auto fill_from_flat = [](xt::xarray<float> &arr, const std::array<float, n * n> &flat)
  {
    arr.resize({n, n});
    for (size_t i_y = 0; i_y < n; ++i_y)
    {
      for (size_t i_x = 0; i_x < n; ++i_x)
      {
        arr(i_y, i_x) = flat[i_y * n + i_x];
      }
    }
  };

  const std::array<float, n * n> plane_flat{
      -5.029797554e+00F, -1.000491619e+00F, -1.781183958e+00F, -4.794006348e+00F, 7.407913685e+00F,
      -6.299201488e+00F, -9.606771469e+00F, 9.065040588e+00F,  3.609016180e+00F,  -2.682374716e-01F,
      9.300536156e+00F,  -2.132025242e+00F, -8.408848763e+00F, -2.971851587e+00F, -6.727296829e+00F,
      9.663336754e+00F,  7.612563610e+00F,  -1.187306345e-01F, -1.980815172e+00F, -9.741707444e-01F,
      4.417536736e+00F,  -5.044634342e+00F, 2.455599070e+00F,  -7.151023865e+00F, -5.976474285e+00F,
      -8.375645638e+00F, 7.000000000e+01F,  -8.885234833e+00F, 1.990729690e+00F,  4.459952831e+00F,
      9.405794144e+00F,  6.431389332e+00F,  5.510213375e-01F,  -6.500000000e+01F, -2.920356035e+00F,
      -8.419393539e+00F, 1.118287563e+00F,  -6.684110641e+00F, -4.095243454e+00F, 6.812169552e+00F,
      -2.748759747e+00F, -3.183720350e+00F, 4.208034992e+00F,  -9.365844727e+00F, -5.522426963e-01F,
      4.631532669e+00F,  5.464315414e+00F,  -5.175104141e+00F, -3.876059055e+00F, -7.267007351e+00F,
      1.914714336e+00F,  -6.748622894e+00F, 6.046544075e+00F,  2.533043735e-03F,  -8.810113907e+00F,
      -2.162684441e+00F, 2.395509958e+00F,  -6.236163139e+00F, -1.442230463e+00F, -8.220367432e+00F,
      4.652608871e+00F,  -2.134012938e+00F, -3.117272139e+00F, -2.062305212e+00F, -9.652559280e+00F,
      -6.327000856e-01F, 7.865010262e+00F,  -6.768711567e+00F, 6.889189243e+00F,  -4.498633862e+00F,
      5.374650478e+00F,  6.680052280e+00F,  8.025926590e+00F,  9.436498642e+00F,  -1.419276714e+00F,
      3.721686125e+00F,  -3.474450350e+00F, 5.949783802e+00F,  -1.309432507e+00F, 2.038878798e-01F,
      5.266572952e+00F,  -6.521616936e+00F, -9.510892630e-01F, 7.963827133e+00F,  3.785662413e+00F,
      -4.280600071e+00F, -1.582730174e+00F, 1.358414888e+00F,  9.352631569e+00F,  1.214209795e+00F,
      9.457283974e+00F,  -6.580185413e+00F, -5.281892776e+00F, 1.325879812e+00F,  3.066298723e+00F,
      9.277424812e+00F,  7.709513187e+00F,  -2.635483027e+00F, -3.163423061e+00F, 7.013430595e+00F,
      -2.226123333e+00F, 3.328154087e+00F,  -1.128790259e+00F, 2.178803682e+00F,  5.688426495e+00F,
      9.359519958e+00F,  -2.534639359e+00F, 3.454485893e+00F,  6.153600693e+00F,  3.949203014e+00F,
      5.500000000e+01F,  -4.659529686e+00F, -3.629975796e+00F, -6.800706863e+00F, 6.321490288e+00F,
      7.455484867e+00F,  6.518468380e+00F,  -8.000000000e+01F, 3.423724473e-01F,  1.053927541e+00F,
      1.162668824e+00F,  9.874370575e+00F,  9.730725288e+00F,  2.036260366e-01F,  5.113427639e+00F,
      8.731904984e+00F,  4.039201736e+00F,  -1.354515076e+00F, -7.559083939e+00F, 2.605735540e+00F,
      5.347581387e+00F,  -4.732100010e+00F, -7.535275459e+00F, 4.453575611e+00F,  -4.612079144e+00F,
      -4.709517956e+00F, -5.973930359e+00F, 7.159346104e+00F,  1.255945802e+00F,  -4.623353958e+00F,
      -4.462284744e-01F, -1.557145834e+00F, 2.160182476e+00F,  4.375789165e+00F};

  const std::array<float, n * n> selfnoise_offset_flat{
      1.386121154e+00F, 1.160266399e+00F, 1.615039110e-01F, 1.149602383e-01F, 5.569685102e-01F, 8.828964829e-02F,
      2.170422792e+00F, 8.066837192e-01F, 2.622750998e+00F, 3.987563252e-01F, 2.408681870e+00F, 1.713569880e+00F,
      2.305953979e+00F, 2.420873642e+00F, 1.053138018e+00F, 5.854074359e-01F, 7.361798286e-01F, 1.218268275e+00F,
      2.490206718e+00F, 2.810428858e+00F, 1.736583710e+00F, 2.409288406e+00F, 1.703367233e+00F, 1.065678716e+00F,
      2.137266159e+00F, 6.519975662e-01F, 6.743201613e-01F, 1.026772976e+00F, 1.098922014e+00F, 1.207523823e+00F,
      1.007071376e+00F, 1.281958342e+00F, 2.568588734e+00F, 1.457903534e-01F, 4.370241165e-01F, 1.946143746e+00F,
      2.376921415e+00F, 3.742421269e-01F, 1.167340398e+00F, 1.240260839e+00F, 1.281449437e+00F, 1.690263271e+00F,
      1.917508721e+00F, 4.422242641e-01F, 6.131253242e-01F, 1.486241817e-01F, 4.034721255e-01F, 2.344391346e+00F,
      1.938493490e+00F, 1.001261473e+00F, 1.257091284e+00F, 2.854349613e+00F, 2.700793028e+00F, 1.283821583e+00F,
      1.582319975e+00F, 2.482690573e+00F, 2.817137957e+00F, 1.600690722e+00F, 1.590818524e+00F, 1.367835760e+00F,
      1.345994830e+00F, 5.039305687e-01F, 1.132501125e+00F, 2.813317537e+00F, 1.790430784e+00F, 1.678036094e+00F,
      2.803580523e+00F, 1.592369914e+00F, 2.344782352e+00F, 1.920359612e+00F, 8.013944142e-03F, 1.508728981e+00F,
      2.064068556e+00F, 1.423973739e-01F, 8.765050769e-01F, 2.054494381e+00F, 2.571492195e+00F, 2.354978770e-01F,
      1.222481728e+00F, 2.747196674e+00F, 9.073323011e-01F, 1.392309308e+00F, 2.146954775e+00F, 8.707171679e-01F,
      2.207285643e+00F, 1.200717330e+00F, 9.387104511e-01F, 1.104948759e+00F, 4.403098822e-01F, 1.702804685e+00F,
      2.519932985e+00F, 1.447441876e-01F, 2.519791365e+00F, 1.948524833e+00F, 1.411289096e+00F, 2.926815748e+00F,
      1.959911942e+00F, 5.211399198e-01F, 7.592728734e-01F, 2.672300100e+00F, 2.116127908e-01F, 1.111235738e+00F,
      1.807494164e+00F, 2.952727079e+00F, 2.754106045e+00F, 8.479128480e-01F, 6.103649735e-02F, 2.139286041e+00F,
      2.908491611e+00F, 4.351080358e-01F, 2.162890509e-02F, 1.940124154e+00F, 2.958271027e+00F, 2.788882017e+00F,
      2.172367334e+00F, 2.641012669e+00F, 6.288098544e-02F, 1.845053315e+00F, 7.991727442e-02F, 2.174944878e+00F,
      1.580471516e+00F, 2.733846903e+00F, 8.258916140e-01F, 1.432026029e-01F, 1.110920012e-01F, 5.675471425e-01F,
      1.961851358e+00F, 2.385084629e+00F, 1.504261851e+00F, 2.195005417e+00F, 2.904832602e+00F, 1.425137758e+00F,
      1.632078648e+00F, 2.927808285e+00F, 2.766167879e+00F, 1.778922200e+00F, 2.623203516e+00F, 1.001332849e-01F,
      5.233525634e-01F, 1.580128431e+00F, 2.095512867e+00F, 1.195294261e+00F, 1.283261061e+00F, 1.508989930e-01F};

  const std::array<float, n * n> dde_noise_flat{
      3.061434031e-01F, 3.269647360e-01F, 1.731094599e+00F, 1.014238715e+00F, 6.236901134e-02F, 1.205987573e+00F,
      1.890144944e+00F, 1.177365780e+00F, 1.188735962e+00F, 8.226304054e-01F, 1.166937351e+00F, 3.581537902e-01F,
      1.250900865e+00F, 1.389739990e+00F, 1.301087618e+00F, 1.517001033e+00F, 1.286719203e+00F, 9.631888270e-01F,
      1.005657077e+00F, 1.024802446e+00F, 8.511241176e-04F, 1.366665244e+00F, 1.552342415e+00F, 1.719320297e+00F,
      1.929298520e+00F, 1.649490833e+00F, 1.479516864e+00F, 1.828801870e+00F, 8.273285627e-01F, 1.597811818e+00F,
      1.802888393e+00F, 1.704037309e+00F, 8.631820679e-01F, 1.388028741e+00F, 8.434457183e-01F, 3.432239592e-01F,
      3.616076112e-01F, 1.733383060e+00F, 3.341195881e-01F, 1.674401760e+00F, 6.000795960e-01F, 1.777386516e-01F,
      1.050287932e-01F, 1.679530740e+00F, 1.559206367e+00F, 1.577399373e-01F, 2.145000361e-02F, 4.590274096e-01F,
      6.235474348e-01F, 4.177598357e-01F, 5.952453017e-01F, 9.210389256e-01F, 1.203354359e+00F, 2.088484541e-02F,
      8.599845767e-01F, 1.112845778e+00F, 1.814174414e+00F, 5.381677747e-01F, 1.720009685e+00F, 1.057121396e+00F,
      2.461522818e-01F, 1.537229776e+00F, 1.927493930e+00F, 1.403891802e+00F, 1.709625006e+00F, 1.271078110e+00F,
      9.762997627e-01F, 1.538764596e+00F, 1.152114511e+00F, 1.794899464e+00F, 7.809233069e-01F, 8.749239445e-01F,
      4.653702676e-01F, 6.379269361e-01F, 6.133252382e-01F, 1.681311242e-02F, 1.012040854e+00F, 7.597048283e-01F,
      1.458170056e+00F, 1.694067717e+00F, 7.335069776e-01F, 5.758321285e-01F, 8.863311410e-01F, 4.760144353e-01F,
      1.289145350e-01F, 1.845118999e+00F, 1.917248845e+00F, 1.651036382e+00F, 1.958101511e+00F, 1.834069014e+00F,
      5.336169004e-01F, 1.742441803e-01F, 1.610710025e+00F, 7.430320978e-01F, 8.032284975e-01F, 1.976903439e+00F,
      1.449887633e+00F, 1.023749948e+00F, 1.796728253e+00F, 1.770944476e+00F, 1.833049655e+00F, 9.386120439e-01F,
      1.543393016e+00F, 6.937389374e-01F, 4.571759701e-01F, 1.121892929e+00F, 1.723438025e+00F, 1.836445332e+00F,
      3.479418457e-01F, 1.933501363e+00F, 7.044675350e-01F, 1.364202499e+00F, 1.502279520e+00F, 4.923846126e-01F,
      1.239266396e+00F, 1.724031210e+00F, 9.928195477e-01F, 1.988111258e+00F, 3.743249476e-01F, 1.753494263e+00F,
      1.496000767e+00F, 8.293555379e-01F, 1.499464869e+00F, 1.028111458e+00F, 1.143773556e+00F, 1.242740989e+00F,
      5.076289177e-01F, 1.529889464e+00F, 3.637962639e-01F, 4.189941287e-01F, 1.714104176e+00F, 6.888084412e-01F,
      1.400167823e+00F, 5.241824389e-01F, 1.265960217e+00F, 1.470629573e+00F, 3.135488033e-01F, 1.790772200e+00F,
      8.536433578e-01F, 1.074633718e+00F, 1.825888753e+00F, 1.323219299e+00F, 1.294353724e+00F, 3.657908365e-02F};

  const std::array<float, n * n> selfnoise_raw_flat{
      3.263707161e+00F,  3.156909943e+00F,  4.359416008e+00F,  -4.748517513e+00F, -1.277578235e+00F,
      4.809598446e+00F,  3.340610504e+00F,  3.995746374e-01F,  -6.222069860e-01F, -4.104931831e+00F,
      -2.690456808e-02F, 3.761811495e+00F,  -3.954714060e+00F, 5.932923555e-01F,  2.278227329e+00F,
      -3.603037596e+00F, -5.032598376e-01F, -1.283097863e+00F, 2.954372883e+00F,  3.577071190e+00F,
      3.733631611e+00F,  1.942152262e+00F,  -9.833062291e-01F, -9.802003503e-01F, -3.375924587e+00F,
      -6.459985375e-01F, -1.508080363e-01F, -2.379989147e+00F, 3.422166586e+00F,  2.486654520e+00F,
      -4.594126225e+00F, 1.484809995e+00F,  -3.835366070e-01F, 2.255144596e+00F,  -3.230953217e+00F,
      -3.348503351e+00F, -4.722969532e+00F, -1.612200856e+00F, -3.678589344e+00F, 2.430459261e+00F,
      -3.095371723e+00F, 4.139674664e+00F,  3.160461903e+00F,  1.786635637e+00F,  4.168058872e+00F,
      -3.736169100e+00F, -4.997955322e+00F, 4.702178955e+00F,  1.083306670e+00F,  4.045416832e+00F,
      1.943035841e+00F,  -1.320744008e-01F, 3.771632433e+00F,  4.824094772e+00F,  1.981672525e+00F,
      3.577477634e-01F,  -4.305338264e-01F, 6.171965599e-01F,  1.082181334e+00F,  2.851767302e+00F,
      3.040936947e+00F,  -1.868041277e+00F, 1.879197478e+00F,  2.891801357e+00F,  -4.240736485e+00F,
      4.958006859e+00F,  7.844354510e-01F,  3.442721844e+00F,  1.758845568e+00F,  2.743127346e-01F,
      4.889481544e+00F,  -4.809790611e+00F, -2.507627487e+00F, 3.059676886e+00F,  -3.780720234e-01F,
      -1.861190438e+00F, -4.650552273e+00F, -3.663688421e+00F, -1.527383924e+00F, -2.707125187e+00F,
      -1.392116904e+00F, 4.120736122e+00F,  -9.670047760e-01F, -4.747276306e+00F, 5.089805126e-01F,
      5.775448680e-01F,  -2.144834518e+00F, 4.203500748e+00F,  -3.008959293e+00F, 1.741747856e+00F,
      5.214340240e-02F,  -3.510877609e+00F, -2.367299557e+00F, -3.822232962e+00F, 2.292112827e+00F,
      4.428717136e+00F,  -2.997974873e+00F, 2.327299833e+00F,  4.744963646e+00F,  1.189669967e+00F,
      3.703534901e-01F,  4.150832653e+00F,  -9.989068508e-01F, -4.756150246e-01F, -2.384079933e+00F,
      -6.039205790e-01F, 4.945163250e+00F,  -2.112460136e+00F, -1.605191112e+00F, -4.564212799e+00F,
      3.185455203e-01F,  -3.575722218e+00F, -3.165786266e+00F, 8.941262364e-01F,  -4.723078251e+00F,
      -4.088789940e+00F, -3.502087593e+00F, -2.617845058e+00F, -3.964036703e-01F, -8.848679066e-01F,
      -4.589323044e+00F, 1.798527837e+00F,  -2.747259140e+00F, 1.076203108e+00F,  -2.972753525e+00F,
      -1.720158815e+00F, -2.857826948e+00F, -2.606396198e+00F, 4.915904999e+00F,  1.089345738e-01F,
      -4.071102142e+00F, 1.723930240e+00F,  -4.037349224e+00F, 1.051942587e+00F,  3.399702787e+00F,
      -6.027887017e-02F, 4.903304100e+00F,  -1.616125584e+00F, 4.032990456e+00F,  -4.351568222e+00F,
      6.808651090e-01F,  2.934888124e+00F,  8.943852186e-01F,  -3.194740057e+00F};

  xt::xarray<float> plane_2d, selfnoise_offset_2d, dde_noise_2d, selfnoise_raw_2d;
  fill_from_flat(plane_2d, plane_flat);
  fill_from_flat(selfnoise_offset_2d, selfnoise_offset_flat);
  fill_from_flat(dde_noise_2d, dde_noise_flat);
  fill_from_flat(selfnoise_raw_2d, selfnoise_raw_flat);

  PlaneBatchItem item;
  item.stokes_cube.resize({1, n, n});
  xt::view(item.stokes_cube, 0) = plane_2d;
  item.weights = xt::xarray<float>({1.F});
  item.nonzero_channels = xt::xarray<bool>({true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({1, n, n});
  group.accept_mask.fill(true);
  group.selfnoise_offset = selfnoise_offset_2d;
  group.dde_noise = dde_noise_2d;
  group.selfnoise_raw = selfnoise_raw_2d;

  BandSlice band{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.3F, .label = "bandA"};

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = true; // first Group 3 test exercising the real windowed-RMS estimator in-line.
  config.boxsize = 5;
  config.inner_boxsize = 1;
  config.ddenoise_scaling_factor = 0.7F;
  config.reject_noise_threshold = 0.F;
  config.threshold = 4.F;
  config.beam_size_enabled = false;

  ProcessPlaneResult result;
  process_band(item, group, band, config, result);

  REQUIRE(result.detections.peak_y.size() == 4);
  const auto find_index = [&](long peak_y, long peak_x)
  {
    for (size_t i = 0; i < result.detections.peak_y.size(); ++i)
    {
      if (result.detections.peak_y[i] == peak_y && result.detections.peak_x[i] == peak_x)
      {
        return i;
      }
    }
    FAIL("no detection at (" << peak_y << ", " << peak_x << ")");
    return static_cast<size_t>(0);
  };

  const auto ratio = config.max_to_std_ratio_value();

  struct ExpectedIsland
  {
    long peak_y, peak_x;
    float flux, local_noise, local_noise_median_ratio, signal_to_noise, signal_to_noise_rms, extreme,
        signal_to_noise_local_mad, signal_to_noise_global_mad, self_noise, excess_noise_factor;
  };

  // Expected values from the independent Python reference (see test file header comment); `extreme` is the
  // postage-stamp max-abs after inner-box zeroing -- signal_to_noise_extended_box's expected value is derived from
  // it via the live config.max_to_std_ratio_value() call below, same pattern as the border-padding test above.
  const std::vector<ExpectedIsland> expected{
      {2, 2, 70.F, 1.521881962e+01F, 1.006362200e+00F, 4.599568367e+00F, 1.384368904e+01F, 8.408848763e+00F,
       9.612234116e+00F, 1.014038944e+01F, -1.508080363e-01F, 1.004153848e+00F},
      {2, 9, -65.F, 1.407065201e+01F, 9.304382801e-01F, 4.619544506e+00F, -1.101443786e+01F, 9.365844727e+00F,
       -7.683967113e+00F, -9.416075706e+00F, 2.255144596e+00F, 1.002945423e+00F},
      {9, 2, 55.F, 1.225867271e+01F, 8.106190562e-01F, 4.486619473e+00F, 1.071099239e+01F, 9.352631569e+00F,
       8.184073448e+00F, 7.967448711e+00F, 3.185455203e-01F, 1.001738787e+00F},
      {9, 9, -80.F, 1.681679916e+01F, 1.112030506e+00F, 4.757147789e+00F, -1.823739351e+01F, 9.277424812e+00F,
       -1.654992867e+01F, -1.158901596e+01F, -2.617845058e+00F, 1.006769300e+00F},
  };

  for (const auto &e : expected)
  {
    const auto i = find_index(e.peak_y, e.peak_x);
    CHECK(result.detections.island_size[i] == 1);
    CHECK(result.detections.local_noise[i] == Catch::Approx(e.local_noise).epsilon(1e-4));
    CHECK(result.detections.local_noise_median_ratio[i] == Catch::Approx(e.local_noise_median_ratio).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise[i] == Catch::Approx(e.signal_to_noise).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_rms[i] == Catch::Approx(e.signal_to_noise_rms).epsilon(1e-4));
    const auto expected_extended_box = e.flux * ratio / e.extreme;
    CHECK(result.detections.signal_to_noise_extended_box[i] == Catch::Approx(expected_extended_box).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_local_mad[i] == Catch::Approx(e.signal_to_noise_local_mad).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_global_mad[i] ==
         Catch::Approx(e.signal_to_noise_global_mad).epsilon(1e-4));
    CHECK(result.detections.self_noise[i] == Catch::Approx(e.self_noise).epsilon(1e-4));
    CHECK(result.detections.excess_noise_factor[i] == Catch::Approx(e.excess_noise_factor).epsilon(1e-4));
    CHECK(result.detections.beam_chi2[i] == 0.F);
    CHECK(result.detections.boost[i] == 0.F);
  }
}
