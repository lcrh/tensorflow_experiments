"""Playing around with keras to read text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.preprocessing.text
import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("input_text", "", "URL or path to input file.")

FLAGS = flags.FLAGS

def get_txt(source):
    if source.startswith("http"):
        print("Downloading file: " + source)
        text = urllib2.urlopen(source).read()
        return text
    else:
        print("Reading file: " + source)
        return open(source, 'r').read()

_tokenizer = None

def singleton(f):
    cache = [None]
    def lazy_f():
        if not cache[0]:
            cache[0] = f()
        return cache[0]
    return lazy_f

@singleton
def tokenizer():
    return keras.preprocessing.text.Tokenizer(
            20000,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\'')

@singleton
def txt():
    return get_txt(FLAGS.input_text)

def tokens():
    tokenizer().fit_on_texts([txt()])
    (seq,) = tokenizer().texts_to_sequences([txt()])
    return seq
 
def main():
    tokens()
    rev_index = {val: key for (key, val) in tokenizer().word_index.iteritems()}
    print([rev_index[i] for i in tokens()])

if __name__ == "__main__":
    main()
