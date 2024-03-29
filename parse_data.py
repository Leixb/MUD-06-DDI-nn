#!/usr/bin/env python3


import sys
from dataset import Dataset

# preprocess a dataset with StanfordCore, and store it in a pickle file for later use
# usage:  ./parse_data.py data-folder filename
#   e.g.  ./parse_data.py ../../data/train train

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ./parse_data.py <data-folder> <filename>")
        sys.exit(1)

    datadir = sys.argv[1]
    filename = sys.argv[2]

    data = Dataset(datadir)
    data.save(filename)

