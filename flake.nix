{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.astro-nix.url = "github:adevress/astro-nix/37460e9a0f35da16525ff8c79400fb51d9aa67b0";

  outputs = { self, nixpkgs, astro-nix }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
      # CUDA 12.6 toolchain (unfree redistributables, the newest the kuma 560 driver supports) from github:adevress/astro-nix (pinned in flake.lock), cached on cache.astro-nix.space
      cuda = astro-nix.legacyPackages.x86_64-linux.cudaPackages_12_6;
      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        pip
        meson-python
        pybind11
        numpy
        xtensor-python
        pytest
      ]);
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = [
          pkgs.meson
          pkgs.ninja
          pkgs.pkg-config
          pkgs.uv
          pythonEnv
          pkgs.xtensor
          pkgs.xtl
          pkgs.xsimd
          pkgs.nlohmann_json
          cuda.cuda_nvcc
          cuda.cuda_cudart
          cuda.cuda_cccl
          cuda.cuda_nvtx
        ];
        shellHook = ''
          export PKG_CONFIG_PATH="${pkgs.python3Packages.numpy}/${pkgs.python3.sitePackages}/numpy/_core/lib/pkgconfig:$PKG_CONFIG_PATH"
          export CC=/usr/bin/gcc
          export CXX=/usr/bin/g++
          # nvcc takes the nix CUDA headers and static runtime by flag, and links with the system ld and glibc.
          export NVCC_APPEND_FLAGS="-Xcompiler -B/usr/bin -I${cuda.cuda_cudart}/include -I${cuda.cuda_cccl}/include -I${cuda.cuda_nvtx.include}/include -L${cuda.cuda_cudart}/lib"
          unset GCC_ROOT
        '';
      };
    };
}