"""Create word2vec embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import urllib2
import re

import numpy as np
import tensorflow as tf

_TXT_URL = "http://www.gutenberg.org/cache/epub/10/pg10.txt"

flags = tf.app.flags
flags.DEFINE_string("input_text", _TXT_URL, "URL or path to input file.")

FLAGS = flags.FLAGS

def get_txt():
    source = FLAGS.input_text
    if source.startswith("http"):
        print("Downloading file from " + source)
        text = urllib2.urlopen(_TXT_URL).read()
        return text
    else:
        print("Using local file: " + source)
        return open(source, 'r').read()

class Word2Vec(object):

    def __init__(self, vec_size, text):
        self._tokens = None
        self._word_dict = None
        self._text = None
        self._raw_text = text
        self._vec_size = vec_size
        self._numeric = np.array(map(
            lambda x: self.word_dict[x], self.tokens))
        self._embedding = tf.Variable(
            np.random.normal(size=((len(self.word_dict),
                                   self._vec_size))))
        self._embedding_w = tf.Variable(
            np.random.normal(size=((len(self.word_dict),
                                   self._vec_size))))
        self._embedding_b = tf.Variable(
            np.random.normal(size=(len(self._word_dict),)))

    def embedding(self, indices):
        """Input tensor integers, return embedding of vec."""
        return tf.nn.embedding_lookup(self._embedding_w, indices, max_norm=1)

    @property
    def text(self):
        if not self._text:
            self._text = re.sub("[^a-zA-Z']+", " ", self._raw_text.lower())
        return self._text
                
    @property
    def tokens(self):
        if not self._tokens:
            self._tokens = re.split(" ", self.text)
        return self._tokens

    @property
    def word_dict(self):
        token_count = {}
        for t in self.tokens:
            if t in token_count:
                token_count[t] += 1
            else:
                token_count[t] = 1
        by_occurrence = sorted((num, w) for w, num in token_count.iteritems())

        if not self._word_dict:
            self._word_dict = {w: i for i, (_, w) in enumerate(self.tokens)}
        return self._word_dict
    
    def batch_loss(self, input_batch):
        target_emb = self.embedding(input_batch[:,0])

    def train_graph(self, n_batches, pre_context, post_context):
        """Create a training graph."""
        raw_input_batch = tf.stack(
            [tf.random_crop(
                self._numeric, (pre_context + post_context + 1,)) for
             _ in xrange(n_batches)])
        # First is target, rest is context.
        input_batch = tf.concat([
            raw_input_batch[pre_context:pre_context + 1],
            raw_input_batch[pre_context],
            raw_input_batch[pre_context + 1:]], 0)

        return input_batch

def main():
    wv = Word2Vec(10, get_txt())
    print(wv.text)
    rand = tf.random_crop([0,1,2,3,4], (2,))
    rand2 = tf.stack([rand, rand])
    with tf.Session() as sess:
        print(sess.run(wv.train_graph(2, 1, 1)))


if __name__ == "__main__":
    main()
