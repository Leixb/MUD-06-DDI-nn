#!/usr/bin/env python3

import sys
import os

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  plot_model.py <fname> [output]")
        sys.exit(1)

    fname = sys.argv[1]
    model = load_model(fname)

    output = sys.argv[2] if len(sys.argv) > 2 else "plots/model.pdf"

    if not os.path.exists("plots"):
        os.mkdir("plots")

    print("Plotting model: {}".format(fname))

    plot_model(model, to_file=output, show_shapes=True)

    print(model)
