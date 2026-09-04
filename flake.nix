{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
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
          pkgs.nlohmann_json
        ];
        shellHook = ''
          export PKG_CONFIG_PATH="${pkgs.python3Packages.numpy}/${pkgs.python3.sitePackages}/numpy/_core/lib/pkgconfig:$PKG_CONFIG_PATH"
          export CC=/usr/bin/gcc
          export CXX=/usr/bin/g++
        '';
      };
    };
}