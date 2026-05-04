{
  description = "Genesis Oracle Dev Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          uv
          git
          python311
          stdenv.cc.cc.lib
          zlib
        ];

        shellHook = ''
          export UV_PYTHON_DOWNLOADS=never
          export UV_PYTHON=python3

          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH
        '';
      };
    };
}
