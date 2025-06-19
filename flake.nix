{
  description = "An empty project that uses Zig.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devshell.url = "github:numtide/devshell";
    zig.url = "github:mitchellh/zig-overlay";
    flake-compat.url = "https://flakehub.com/f/edolstra/flake-compat/1.tar.gz";
  };

  outputs =
    { self, ... }@inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.devshell.flakeModule ];

      systems = builtins.attrNames inputs.zig.packages;

      perSystem =
        { system, pkgs, ... }:
        {
          devshells.default = {
            # zls is only works with 0.14.0 for now
            commands = [ { package = inputs.zig.packages.${system}."0.14.0"; } ];

            packages = [ pkgs.zls ];
          };
        };
    };
}
