"""Train an LSTM on text file."""

import urllib2

import numpy as np
import tensorflow as tf

_TXT_URL = "http://www.gutenberg.org/cache/epub/10/pg10.txt"

def get_training_data():
    text = urllib2.urlopen(_TXT_URL).read()
    return np.array([ord(c) for c in text])

def get_pred_model():
    input_ = Input(shape=(256,))

def main():
    data = get_training_data()

if __name__ == "__main__":
    main()
