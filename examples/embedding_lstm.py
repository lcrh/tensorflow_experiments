"""Attempt to learn a semantic embedding as a side effect of text prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import keras
import keras.preprocessing.text
import numpy as np
import tensorflow as tf

_TXT_PATH = '/home/leo/Desktop/pg10.txt'
_STOP_WORD_PATH = 'stop_words.txt'
_LEN_LIMIT = 2600
_WORD_NUM_LIMIT = 10
_MIN_WORD_RANK=3000
_EMBED_DIM = 30
_N_STATES = 512

full_text = open(_TXT_PATH, 'r').read().replace('\r', '')
all_paragraphs = full_text.split('\n\n')

stop_words = open(_STOP_WORD_PATH, 'r').read().split('\r\n')[:-1]

max_len = min(max(len(s) for s in all_paragraphs), _LEN_LIMIT)

print('Setting training sequence length to {}'.format(max_len))
paragraphs = [s for s in all_paragraphs if len(s) <= max_len]

discarded = len(all_paragraphs) - len(paragraphs)
print('Discarded {} paragraphs that were too long.'.format(discarded))

print('Preparing numpy input array.')
inp_arr = np.zeros(shape=(len(paragraphs), max_len))

for i, s in enumerate(paragraphs):
    inp_arr[i, 0:len(s)] = [ord(c) for c in s]

print('Tokenizing inputs')
# Remove stopwords from text.
stop_word_re = r'\b(' + r'|'.join(stop_words) + r'|[0-9]*:[0-9]*)\b'
full_text_no_stopwords = re.sub(stop_word_re, '', full_text.lower())
tokenizer = keras.preprocessing.text.Tokenizer(_MIN_WORD_RANK)
tokenizer.fit_on_texts([full_text_no_stopwords])

# Indices correspond to paragraphs.
word_tokens = [set(w) for w in tokenizer.texts_to_sequences(paragraphs)]

# Keep only "WORD_NUM_LIMIT" most unusual words and pad with zeros.
word_arrs = np.zeros(shape=(len(paragraphs), _WORD_NUM_LIMIT))
for i, w in enumerate(word_tokens):
    words = sorted(w)[-_WORD_NUM_LIMIT:]
    word_arrs[i, 0:len(words)] = words

print('Building training graph')
# Create tensor flow input arrays.
tf_input_text = tf.constant(inp_arr, dt=tf.uint8)
tf_input_words = tf.constant(word_arrs, dt=tf.uint16)

# Create embedding vars.
embedding_var = tf.Variable(
        np.random.random((_MIN_WORD_RANK, _EMBED_DIM))*2 - 1,
        dtype=tf.float32)

# Create lstm cell
lstm = tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(_N_STATES), state_keep_prob=0.95)

