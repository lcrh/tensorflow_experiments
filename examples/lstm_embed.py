"""Train an LSTM to find a sentence embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import urllib2

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "source", metavar="URL", type=str,
            help="Input text location, e.g., %s" % _TXT_URL,
            default=_TXT_URL)
    return parser

FLAGS = get_parser().parse_args()

def get_txt():
    source = FLAGS.source
    if source.startswith("http"):
        print("Downloading file from " + source)
        text = urllib2.urlopen(_TXT_URL).read()
        return text
    else:
        print("Using local file: " + source)
        return open(source, 'r').read()

def create_lstm(states):
    tf.contrib.rnn.BasicLSTMCell(_N_STATES)

def build_graph(input_sequence, n_states, embedding_dim):
    """Returns output tensor of same shape as input tensor.
    
    Args:
      input_sequence: batch_size x seq_length x input_dim.
      embedding_dim: Number of embedding dimensions.
    """
    out_logits, state = tf.nn.static_rnn(
        create_lstm(n_states),
        input_char_list,
        dtype=tf.float32)

if __name__ == "__main__":
    main()
