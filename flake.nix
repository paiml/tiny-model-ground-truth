{
  description = "tiny-model-ground-truth - Popperian falsification for model parity";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311
            uv
            rustup
            gnumake
          ];

          shellHook = ''
            echo "tiny-model-ground-truth dev shell"
            echo "Run: make pull && make convert && make check"
          '';
        };
      });
}
