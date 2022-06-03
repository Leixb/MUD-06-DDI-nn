{
  description = "Python environment managed with poetry and flakes";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, ...}:

  flake-utils.lib.eachDefaultSystem (system:
  let
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };

    stanford_corenlp = pkgs.callPackage ./stanford-corenlp.nix { };

    NLTK_DATA = pkgs.callPackage ./nltk_data.nix { collection = "all"; };

    python = pkgs.python38;

    overrides = let addBuildInput = { drv, self, super, input }:
        super.${drv}.overridePythonAttrs ( old: { nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ self.${input} ]; });
      in pkgs.poetry2nix.overrides.withDefaults (self: super: {
        traitlets = addBuildInput { drv = "traitlets"; input = "hatchling"; inherit self super; };
        terminado = addBuildInput { drv = "terminado"; input = "hatchling"; inherit self super; };
        tensorflow-io-gcs-filesystem = super.tensorflow-io-gcs-filesystem.overridePythonAttrs ( old: { buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.libtensorflow ]; });
        tensorflow = addBuildInput { drv = "tensorflow"; input = "wheel"; inherit self super; };
        tensorflow-gpu = addBuildInput { drv = "tensorflow-gpu"; input = "wheel"; inherit self super; };
        astunparse = addBuildInput { drv = "astunparse"; input = "wheel"; inherit self super; };

    });


    pyEnv = pkgs.poetry2nix.mkPoetryEnv {
      projectDir = ./.;

      inherit python overrides;
      preferWheels = true;

      editablePackageSources = {
        ddi = ./.;
      };
    };

  in
  {
    devShell = pkgs.mkShell {
      name = "python NLTK coreNLP TF";

      STANFORDDIR = "${stanford_corenlp}/share/java";

      inherit NLTK_DATA;

      LD_LIBRARY_PATH = pkgs.lib.strings.makeLibraryPath (with pkgs.cudaPackages; [
        pkgs.linuxPackages_latest.nvidia_x11
        cudatoolkit
        cudnn
        libcublas
        libcufft
        libcurand
        libcusolver
        libcusparse
      ]);

      buildInputs = [ pkgs.cudaPackages.cudatoolkit pyEnv pkgs.jre ];
    };
  });
}
