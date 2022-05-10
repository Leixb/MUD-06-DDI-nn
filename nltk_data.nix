{ lib
, stdenvNoCC
, fetchFromGitHub
, bash
, python2
, unzip
, collection ? "all"
}:

# Manually add nltk_data and put it in the /nix/store
# This guarantees that the data is both immutable and
# the same in all systems.
stdenvNoCC.mkDerivation rec {
  pname = "nltk_data";
  version = "444941def3726b7d12605b5cfdbf2f906af152a9";

  src = fetchFromGitHub {
    owner = "nltk";
    repo = pname;
    rev = version;
    sha256 = "sha256-Y+J6Ddd1h1A3y0JwLorKIMGIIczWasfqBa1HX7mvveQ=";
  };

  dontConfigure = true;
  dontBuild = true;

  nativeBuildInputs = [ python2 unzip ];

  # Despite the name, download.sh does not download anything from the
  # internet
  installPhase = ''
    export NLTK_DATA_DIR=$out
    bash tools/download.sh "${collection}"
  '';
}
