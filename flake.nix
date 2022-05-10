{
  description = "Python environment managed with mach-nix and flakes";

  inputs = {
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    nixpkgs = {
      url = "github:numtide/nixpkgs-unfree";
      inputs.nixpkgs.follows = "nixpkgs-unstable";
    };

    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # use `nix flake lock --update-input pypi-deps-db` to update the pypi database
    # or `nix flake update` to update all
    pypi-deps-db = {
      url = "github:DavHau/pypi-deps-db";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.mach-nix.follows = "mach-nix";
    };

    mach-nix = {
      url = "github:bjornfor/mach-nix/adapt-to-make-binary-wrapper";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
      inputs.pypi-deps-db.follows = "pypi-deps-db";
    };

    # nixConfig.extra-substituters = [ "https://nixpkgs-unfree.cachix.org" ];
    # nixConfig.extra-trusted-public-keys = [ "nixpkgs-unfree.cachix.org-1:hqvoInulhbV4nJ9yJOEr+4wxhDV4xq2d1DK7S6Nj6rs=" ];

  };
  outputs = { nixpkgs, flake-utils, mach-nix, ...}:

  flake-utils.lib.eachDefaultSystem (system:
  let
    pkgs = nixpkgs.legacyPackages.${system};

    # Do NOT use import mach-nix {inherit system;};
    #
    # otherwise mach-nix will not use flakes and pypi-deps-db
    # input will not be used:
    # https://github.com/DavHau/mach-nix/issues/269#issuecomment-841824763
    mach = mach-nix.lib.${system};

    python-env = mach.mkPython {
      # Choose python version
      python = "python38";

      providers.tensorflow = "nixpkgs";

      overridesPost = [(python-self: python-super: {
        tensorflow = python-super.tensorflow-bin.override {
          cudaSupport = true;
          # cudaPackages = with pkgs.cudaPackages; [
          #   cudnn
          #   cudatoolkit
          #   cuda_nvcc
          #   cuda_cudart
          # ];
        };
      })];


      # Specify python requirements, you can use ./requirements.txt a
      # string (or a combination of both)
      requirements = ''
        ipython
        black
        jupyterlab==4.0.0a24
        bleach==4.1.0
        python-lsp-server
        python-language-server[all]
      '' + (builtins.readFile ./requirements.txt);
    };

  in
  {
    devShell = pkgs.mkShell {
      name = "python";

      LD_LIBRARY_PATH = "${pkgs.linuxPackages.nvidia_x11}/lib";

      buildInputs = [ python-env ];
    };
  });
}
