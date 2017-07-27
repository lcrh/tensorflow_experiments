"""Create word2vec embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cmd
import collections
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

    def __init__(self, embed_dims, text):
        self._tokens = None
        self._word_dict = None
        self._rev_dict = None
        self._text = None
        self._raw_text = text
        self._embed_dims = embed_dims
        self._numeric = np.array(map(
            lambda x: self.word_dict[x], self.tokens), dtype=np.int64)

        self._embedding = tf.Variable(
            np.random.normal(size=((len(self.word_dict),
                                   self._embed_dims))),
            dtype=tf.float32)

        self._final_embedding = tf.Variable(
            np.random.normal(size=((len(self.word_dict),
                                   self._embed_dims))),
            dtype=tf.float32)

        self._embedding_w = tf.Variable(
            np.random.normal(size=((len(self.word_dict),
                                   self._embed_dims))),
            dtype=tf.float32)
        self._embedding_b = tf.Variable(
            np.random.normal(size=(len(self._word_dict),)),
            dtype=tf.float32)
        self._n_samples = 10

    def embedding(self, indices):
        """Input tensor integers, return non-trainable embedding of vec."""
        return tf.nn.embedding_lookup(self._final_embedding,
                                      indices, max_norm=1)

    def weight(self, indices):
        """Input tensor integers, return softmax weights of vecs."""
        return tf.nn.embedding_lookup(self._embedding_w, indices)

    def bias(self, indices):
        """Input tensor integers, return softmax biases of vecs."""
        return tf.nn.embedding_lookup(self._embedding_b, indices)

    @property
    def text(self):
        if not self._text:
            self._text = re.sub("[^a-zA-Z']+", " ", self._raw_text.lower())
        return self._text
                
    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = np.array(re.split(" ", self.text))
        return self._tokens

    @property
    def rev_dict(self):
        """Mapping from index to word."""
        if self._rev_dict is None:
            tokens, counts = np.unique(self.tokens, return_counts=True)
            self._rev_dict = tokens[np.argsort(counts)][::-1]
        return self._rev_dict

    @property
    def word_dict(self):
        if not self._word_dict:
            print("Preparing dictionary.")
            self._word_dict = {w: i for i, w in enumerate(self.rev_dict)}
        return self._word_dict

    def batch_loss(self, input_batch):
        """Compute loss function."""
        batch_size = input_batch.get_shape()[0].value
        dims = self._embed_dims
        num_classes = len(self._word_dict)
        num_true = input_batch.get_shape()[1].value - 1

        # (batch_size x embed_dims)
        target_embs = self.embedding(input_batch[:,0])

        return tf.reduce_sum(tf.nn.nce_loss(
            tf.reshape(self._embedding_w, (num_classes, dims)),
            tf.reshape(self._embedding_b, (num_classes,)),
            tf.reshape(input_batch[:,1:], (batch_size, num_true)),
            target_embs,
            self._n_samples,
            len(self._word_dict),
            num_true=num_true))

    def average_loss(self, batch_size, pre_context, post_context):
        """Create a training graph."""
        raw_input_batch = tf.stack(
            [tf.random_crop(
                self._numeric, (pre_context + post_context + 1,)) for
             _ in xrange(batch_size)])

        # First is target, rest is context.
        input_batch = tf.concat([
            raw_input_batch[pre_context:pre_context + 1],
            raw_input_batch[:pre_context],
            raw_input_batch[pre_context + 1:]], 0)

        return self.batch_loss(input_batch) / input_batch.get_shape()[0].value

    def optimize_op(self, batch_size, pre_context, post_context, optimizer=None):
        """Return optimize op that also updates final embedding."""
        optimizer = optimizer or tf.train.GradientDescentOptimizer(1)

        # Copy over embedding value.
        update_embedding = tf.assign(
            self._final_embedding, self._embedding)

        loss = self.average_loss(batch_size, pre_context, post_context)
        print_loss = tf.Print(loss, [loss], "w2v loss: ")

        # Apply optimization and update final embedding value.
        return tf.group(
            optimizer.minimize(print_loss),
            tf.Print(update_embedding, [loss], "loss: "))

    def train(self, epochs, batch_size,
              pre_context, post_context, optimizer=None):
        """Train model for a number of epochs. Returns final embedding value."""
        opt_op = self.optimize_op(
            batch_size, pre_context, post_context, optimizer)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in xrange(epochs):
                print("w2v training batch %d" % i)
                sess.run(opt_op)
            result = sess.run(self._embedding)
        return result

    def set_embedding(self, value):
        """Return a new operation that sets the provided embedding value."""
        return tf.assign(self._final_embedding, tf.constant(value))

    def word_similarity(self, embedding, word1, word2):
        """Compute cosine dist of two words using a concrete embedding."""
        index1, index = (self.word_dict[word1.lower()],
                         self.word_dict[word2.lower()])
        embed1, embed2 = embedding[index1], embedding[index2]
        cosine_sim = numpy.dot(embed1, embed2) / (
            numpy.power(numpy.sum(numpy.power(embed1, 2)), 0.5), + 
            numpy.power(numpy.sum(numpy.power(embed2, 2)), 0.5))
        return consine_sim

    def similarities(self, embeddings, word):
        """Return array [[word_index, sim], ...] by decreasing similarity."""
        word_index = self.word_dict[word.lower()]
        word_embedding = embeddings[word_index]
        word_norm = np.linalg.norm(word_embedding)

        norms = np.linalg.norm(embeddings, axis=1)
        similarities = (
            np.dot(embeddings, word_embedding) / (
                norms * word_norm))

        sorted_indices = np.argsort(similarities)[::-1]
        dtype = [('word', 'S20'), ('similarity', 'f4')]
        result = np.zeros(shape=(len(sorted_indices,)), dtype=dtype)
        print(sorted_indices.shape)
        print(self.rev_dict.shape)
        result['word'] = self.rev_dict[sorted_indices]
        result['similarity'] = similarities[sorted_indices]
        return result

def main():
    wv = Word2Vec(50, get_txt())
    embed = wv.train(1000, 250, 6, 6)
    print(wv.similarities(embed, "king")[:10])
    print(wv.similarities(embed, "dog")[:10])
    print(wv.similarities(embed, "woman")[:10])

if __name__ == "__main__":
    main()
