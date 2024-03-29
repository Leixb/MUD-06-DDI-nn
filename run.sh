#!/usr/bin/env bash

BASEDIR=../resources/DDI

UTIL=./util/

export PYTHONPATH="$UTIL" #Directory of the DDI data

set -e # Abort if something fails

if [[ "$#" == "0" ]]; then
    # if no arguments are given, run all steps
    set -- parse train predict test
fi

if [[ "$*" == *"parse"* ]]; then
   "$UTIL"/corenlp-server.sh -quiet true -port 9000 -timeout 15000 &
   sleep 1

   python3 parse_data.py "$BASEDIR"/data/train train.pck
   python3 parse_data.py "$BASEDIR"/data/devel devel.pck
   python3 parse_data.py "$BASEDIR"/data/test test.pck
   kill "$(cat /tmp/corenlp-server.running)"
fi

if [[ "$*" == *"train"* ]]; then
    rm -rf model*
    python3 train.py train.pck devel.pck model
fi

if [[ "$*" == *"plot"* ]]; then
    python3 plot_model.py model
fi

if [[ "$*" == *"predict"* ]]; then
   rm -f devel.stats devel.out
   python3 predict.py model devel.pck devel.out
   python3 "$UTIL"/evaluator.py DDI "$BASEDIR"/data/devel devel.out | tee devel.stats
fi

if [[ "$*" == *"test"* ]]; then
   rm -f test.stats test.out
   python3 predict.py model test.pck test.out
   python3 "$UTIL"/evaluator.py DDI "$BASEDIR"/data/test test.out | tee test.stats

   UNCOMMITED=false
   git diff-index --quiet HEAD -- || {
      echo "WARNING: There are uncommited changes"
      UNCOMMITED=true
   }
   COMMIT="$(git rev-parse HEAD)"
   PERCENTAGE="$(grep "M.avg" test.stats | cut -f8)"

   echo "$COMMIT,$PERCENTAGE,$UNCOMMITED" >> results.csv
fi

