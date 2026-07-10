{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = [
          pkgs.meson
          pkgs.ninja
          pkgs.pkg-config
          pkgs.python3
          pkgs.python3Packages.pip
          pkgs.python3Packages.meson-python
          pkgs.python3Packages.pybind11
          pkgs.python3Packages.numpy
          pkgs.python3Packages.xtensor-python
          pkgs.xtensor
          pkgs.xtl
          pkgs.tbb
        ];
        # numpy's numpy.pc lives inside site-packages, off pkg-config's default
        # search path; numpy-config --pkgconfigdir points at exactly this
        # directory for exactly this purpose, so dependency('numpy') in
        # meson.build works with no special-casing.
        shellHook = ''
          export PKG_CONFIG_PATH="${pkgs.python3Packages.numpy}/${pkgs.python3.sitePackages}/numpy/_core/lib/pkgconfig:$PKG_CONFIG_PATH"
        '';
      };
    };
}