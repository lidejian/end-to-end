# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import codecs
import numpy as np

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

def _sentence_to_word_ids(sentence, word_to_id):
  word_list = sentence.split()
  return [word_to_id[word] for word in word_list if word in word_to_id]

def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "train/sentence")
  valid_path = os.path.join(data_path, "dev/sentence")
  test_path = os.path.join(data_path, "test/sentence")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)

  return train_data, valid_data, test_data, vocabulary


def load_labels(data_path=None):
  with codecs.open("%s/label" % data_path, encoding="utf-8") as fin_label:
    labels = [int(x.strip()) for x in fin_label]
  return labels


def load_data_and_labels(data_path=None):

    with \
            codecs.open("%s/sentence" % data_path, encoding="utf-8") as fin_sentense, \
            codecs.open("%s/label" % data_path, encoding="utf-8") as fin_label:

      word_to_id = _build_vocab("%s/sentence"%data_path)
      sentense = [line.strip() for line in fin_sentense]
      sentence = []
      for sen in sentense:
        sentence.append(_sentence_to_word_ids(sen,word_to_id))

      labels = [int(x.strip()) for x in fin_label]
      num_classes = max(labels) + 1

      _labels = []
      for label in labels:
        l = [0] * num_classes
        l[label] = 1
        _labels.append(l)
      labels = np.array(_labels)
      return sentense, labels

