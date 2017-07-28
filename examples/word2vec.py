"""Create word2vec embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cmd
import collections
import exceptions
import urllib2
import re
import readline

import numpy as np
import tensorflow as tf

_TXT_URL = "http://www.gutenberg.org/cache/epub/10/pg10.txt"

flags = tf.app.flags
flags.DEFINE_string("input_text", _TXT_URL, "URL or path to input file.")
flags.DEFINE_string("word_list", None, "A list of words to use.")
flags.DEFINE_string("stop_word_list", None, "A list of stop words.")
flags.DEFINE_float("omit_percentile", 0.75, "Omit rare words.")

FLAGS = flags.FLAGS

def get_txt(source):
    if source.startswith("http"):
        print("Downloading file: " + source)
        text = urllib2.urlopen(_TXT_URL).read()
        return text
    else:
        print("Reading file: " + source)
        return open(source, 'r').read()

def as_words(text):
    return re.findall("[a-z']+", text.lower())

def preprocess_text(text):
    print("Preprocessing text.")
    word_list = None
    stop_words = None
    if FLAGS.word_list:
        print("Downloading word list.")
        word_list = set(as_words(get_txt(FLAGS.word_list)))
    if FLAGS.stop_word_list:
        print("Downloading stop word list.")
        stop_words = set(as_words(get_txt(FLAGS.stop_word_list)))
    words, counts = np.unique(as_words(text), return_counts=True)
    # Remove rare words.
    words = words[np.argsort(counts)[::-1]][
        :int(FLAGS.omit_percentile * len(words))]
    
    if word_list:
        if stop_words:
            word_list -= stop_words
        return np.array(filter(lambda x: x in word_list, as_words(text)))
    else:
        return np.array(as_words(text))

class Word2Vec(object):

    def __init__(self, embed_dims, text):
        self._tokens = None
        self._word_dict = None
        self._rev_dict = None
        self._text = text
        self._embed_dims = embed_dims

        self._numeric = np.array(map(
            lambda x: self.word_dict[x], self.tokens), dtype=np.int64)

        self._embedding = tf.Variable(
            np.random.random(size=((len(self.word_dict),
                                    self._embed_dims))) * 2 - 1,
            dtype=tf.float32)

        self._final_embedding = tf.Variable(
            np.random.random(size=((len(self.word_dict),
                                    self._embed_dims))) * 2 - 1,
            dtype=tf.float32)

        self._embedding_w = tf.Variable(
            np.random.random(size=((len(self.word_dict),
                                    self._embed_dims))) * 2 - 1,
            dtype=tf.float32)

        self._embedding_b = tf.Variable(
            np.zeros(shape=(len(self._word_dict),)),
            dtype=tf.float32)
        self._n_samples = 1000

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
        return self._text
                
    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = preprocess_text(self.text)
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
        target_embs = tf.nn.embedding_lookup(
                self._embedding, input_batch[:,0])

        return tf.reduce_mean(tf.nn.nce_loss(
            tf.reshape(self._embedding_w, (num_classes, dims)),
            tf.reshape(self._embedding_b, (num_classes,)),
            tf.reshape(input_batch[:,1:], (batch_size, num_true)),
            target_embs,
            self._n_samples,
            len(self._word_dict),
            num_true=num_true,
            remove_accidental_hits=True))

    def average_loss(self, batch_size, pre_context, post_context):
        """Create a training graph."""
        raw_input_batch = tf.stack(
            [tf.random_crop(
                self._numeric, (pre_context + post_context + 1,)) for
             _ in xrange(batch_size)], name="raw_batch")

        # First is target, rest is context.
        input_batch = tf.concat([
            raw_input_batch[:, pre_context:pre_context + 1],
            raw_input_batch[:, :pre_context],
            raw_input_batch[:, pre_context + 1:]], 1,
            name="input_batch")

        return self.batch_loss(input_batch)

    def optimize_op(self, batch_size, pre_context, post_context, optimizer=None):
        """Return optimize op that also updates final embedding."""
        global_step = tf.train.create_global_step()
        advance_step = tf.assign_add(global_step, 1)

        if not optimizer:
            learning_rate = tf.train.inverse_time_decay(
                0.15, global_step, 100000, decay_rate=100)
            learning_rate = tf.Print(learning_rate, [learning_rate], "alpha: ")
            optimizer = tf.train.RMSPropOptimizer(learning_rate)

        # Copy over embedding value.
        update_embedding = tf.assign(
            self._final_embedding, self._embedding)

        loss = self.average_loss(batch_size, pre_context, post_context)
        print_loss = tf.Print(loss, [loss], "w2v loss: ")

        # Apply optimization and update final embedding value.
        return tf.group(
            advance_step,
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
                print("batch nr %d" % i)
                sess.run(opt_op)
            result = sess.run(self._embedding)
        return result

    def set_embedding(self, value):
        """Return a new operation that sets the provided embedding value."""
        return tf.assign(self._final_embedding, tf.constant(value))

    def embedding_value(self, embeddings, word):
        word_index = self.word_dict[word.lower()]
        return embeddings[word_index]

    def closest_words(self, embeddings, inp, min_perc=0.0, max_perc=1.0):
        """Return array [[word_index, sim], ...] by decreasing similarity."""
        if isinstance(inp, str):
            embedding = self.embedding_value(embeddings, inp)
        else:
            embedding = inp

        norm = np.linalg.norm(embedding)
        norms = np.linalg.norm(embeddings, axis=1)
        norm_mult = norms * norm
        similarities =  np.dot(embeddings, embedding) / (
            np.where(norm_mult <= 0, 1e15, norm_mult))

        shortened_dict = self.rev_dict[
            int(min_perc * len(self.rev_dict)):
            int(max_perc * len(self.rev_dict))]
        similarities = similarities[
            int(min_perc * len(similarities)):
            int(max_perc * len(similarities))]

        sorted_indices = np.argsort(similarities)[::-1]
        dtype = [('word', 'S20'), ('similarity', 'f4')]
        result = np.zeros(shape=(len(sorted_indices,)), dtype=dtype)
        result['word'] = shortened_dict[sorted_indices]
        result['similarity'] = similarities[sorted_indices]
        return result

class Shell(cmd.Cmd, object):

    def __init__(self, wv, embeddings):
        super(Shell, self).__init__()
        self._wv = wv
        self._emb = embeddings / np.linalg.norm(embeddings, axis=1).reshape(
            len(embeddings), 1)

    def do_dist(self, args):
        """Print distance between word1 and word2."""
        try:
            word1, word2 = args.split(" ")
            e1 = self._wv.embedding_value(self._emb, word1)
            e2 = self._wv.embedding_value(self._emb, word2)
            print(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
        except Exception as e:
            print("Error:", e)

    def do_close(self, word):
        """Print words close to word."""
        try:
            print(self._wv.closest_words(self._emb, word)[0:30])
        except Exception as e:
            print("Error:", e)

    def do_add(self, args):
        """Add words."""
        try:
            word1, word2 = args.split(" ")
            e1 = self._wv.embedding_value(self._emb, word1)
            e2 = self._wv.embedding_value(self._emb, word2)
            print(self._wv.closest_words(self._emb, e1 + e2)[0:30])
        except Exception as e:
            print("Error:", e)

    def do_sub(self, args):
        """Subtract words."""
        try:
            word1, word2 = args.split(" ")
            e1 = self._wv.embedding_value(self._emb, word1)
            e2 = self._wv.embedding_value(self._emb, word2)
            print(self._wv.closest_words(self._emb, e1 - e2)[0:30])
        except Exception as e:
            print("Error:", e)

    def do_analogy(self, args):
        """word1 is to word2 as word3 is to ?"""
        try:
            word1, word2, word3 = args.split(" ")
            e1 = self._wv.embedding_value(self._emb, word1)
            e2 = self._wv.embedding_value(self._emb, word2)
            e3 = self._wv.embedding_value(self._emb, word3)
            norm = np.linalg.norm
            target = e3 + (e2 - e1)
            print(self._wv.closest_words(self._emb, target)[0:30])
        except Exception as e:
            print("Error:", e)

    def do_normed_analogy(self, args):
        """word1 is to word2 as word3 is to ?"""
        try:
            word1, word2, word3 = args.split(" ")
            e1 = self._wv.embedding_value(self._emb, word1)
            e2 = self._wv.embedding_value(self._emb, word2)
            e3 = self._wv.embedding_value(self._emb, word3)
            norm = np.linalg.norm
            target = e3 + (e2 - e1) / norm(e2 - e1)
            print(self._wv.closest_words(self._emb, target)[0:30])
        except Exception as e:
            print("Error:", e)

def main():
    wv = Word2Vec(750, get_txt(FLAGS.input_text))
    embed = wv.train(10000, 150, 5, 2)
    Shell(wv, embed).cmdloop()

if __name__ == "__main__":
    main()
