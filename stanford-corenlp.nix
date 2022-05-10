{ lib
, stdenv
, fetchzip
}:

stdenv.mkDerivation rec {
  pname = "stanford-corenlp";
  version = "4.4.0";

  src = fetchzip {
    url = "https://huggingface.co/stanfordnlp/CoreNLP/resolve/v${version}/stanford-corenlp-latest.zip";
    sha256 = "sha256-J9k+RRpK8SsLuckhtdRdRkOz/+KOQospnKfOj0KvxWo=";
  };

  doBuild = false;

  buildPhase = ":";

  installPhase = ''
    runHook preInstall

    mkdir -p $out/share/java
    cp -r . $out/share/java/

    runHook postInstall
  '';
}

