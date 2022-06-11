#!/usr/bin/env python3


import sys
import os
import random
from contextlib import redirect_stdout

import tensorflow as tf
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras import regularizers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Dropout,
    Conv1D,
    MaxPool1D,
    Reshape,
    Concatenate,
    Flatten,
    Bidirectional,
    LSTM,
)

import numpy as np

from dataset import Dataset
from codemaps import Codemaps
from transformer import TokenAndPositionEmbedding, TransformerBlock


def load_glove_embedding(word2index: dict, embedding_dim: int = 100) -> Embedding:
    glove_path = f"../glove.6B/glove.6B.{embedding_dim}d.txt"

    n_words = len(word2index)
    embedding_matrix = np.zeros((n_words, embedding_dim))

    with open(glove_path, "r") as f:
        for _line in f:
            line = _line.split()
            word = line[0]
            if word in word2index:
                idx = word2index[word]
                embedding_vector = np.array(line[1:], dtype=np.float32)
                embedding_matrix[idx] = embedding_vector

    return Embedding(n_words, embedding_dim, weights=[embedding_matrix], trainable=False)


def build_network(codes):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()
    embedding_dim = 100

    # inputs

    inputs = list(map(lambda x: Input(shape=(max_len,), name=f"input_{x}"), ["w", "lw", "rel", "l", "p"]))

    # embeddings

    embeddings = list(map(TokenAndPositionEmbedding(maxlen=max_len, vocab_size=n_words, embed_dim=embedding_dim), inputs))
    embeddings += [load_glove_embedding(codes.word_index, embedding_dim=embedding_dim)(inputs[0])]
    embeddings += [load_glove_embedding(codes.lc_word_index, embedding_dim=embedding_dim)(inputs[1])]

    embeddings = list(map(Dropout(0.1), embeddings))

    # concatenate
    concatenated = Concatenate()(embeddings)

    lstm = Bidirectional(LSTM(units=128, return_sequences=True))(concatenated)

    flat = Flatten()(lstm)

    # dense layers

    dense1 = Dense(n_labels * 4, activation="relu")(flat)

    out = Dense(n_labels, activation="softmax")(dense1)

    model = Model(inputs, out)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  train.py ../data/Train ../data/Devel  modelname
# --

# --------- MAIN PROGRAM -----------
# --
# -- Usage:  train.py ../data/Train ../data/Devel  modelname
# --

def set_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if __name__ == "__main__":

    set_memory_growth()

    if len(sys.argv) != 4:
        print("Usage: train.py <trainfile> <validationfile>  <modelname>")
        exit(1)

    set_random_seed(2795991)
    os.environ['PYTHONHASHSEED'] = str(0)

    # directory with files to process
    trainfile = sys.argv[1]
    validationfile = sys.argv[2]
    modelname = sys.argv[3]

    # load train and validation data
    traindata = Dataset(trainfile)
    valdata = Dataset(validationfile)

    # create indexes from training data
    max_len = 150
    suf_len = 5
    codes = Codemaps(traindata, max_len)

    # build network
    model = build_network(codes)
    with redirect_stdout(sys.stderr):
        model.summary()

    # encode datasets
    Xt = codes.encode_words(traindata)
    Yt = codes.encode_labels(traindata)
    Xv = codes.encode_words(valdata)
    Yv = codes.encode_labels(valdata)

    # train model
    with redirect_stdout(sys.stderr):
        model.fit(Xt, Yt, batch_size=32, epochs=10, validation_data=(Xv, Yv), verbose=1)

    # save model and indexs
    model.save(modelname)
    codes.save(modelname)
