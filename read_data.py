# -*- coding:utf-8 -*-
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

def _sentence_to_word_ids(sentence, word_to_id):
  word_list = sentence.split()
  return [word_to_id[word] for word in word_list if word in word_to_id]

def _trans_label(labels, num_classes):
    _labels = []
    for label in labels:
        l = [0] * num_classes
        l[label] = 1
        _labels.append(l)
    labels = np.array(_labels)
    return labels


def raw_data(data_path=None):
    train_sen_path = os.path.join(data_path, "train/sentence")
    dev_sen_path = os.path.join(data_path, "dev/sentence")
    test_sen_path = os.path.join(data_path, "test/sentence")

    train_lab_path = os.path.join(data_path, "train/label")
    dev_lab_path = os.path.join(data_path, "dev/label")
    test_lab_path = os.path.join(data_path, "test/label")

    with \
            codecs.open(train_sen_path, encoding="utf-8") as fin_train_sen, \
            codecs.open(dev_sen_path, encoding="utf-8") as fin_dev_sen, \
            codecs.open(test_sen_path, encoding="utf-8") as fin_test_sen, \
            codecs.open(train_lab_path, encoding="utf-8") as fin_train_label, \
            codecs.open(dev_lab_path, encoding="utf-8") as fin_dev_label, \
            codecs.open(test_lab_path, encoding="utf-8") as fin_test_label:

        word_to_id = _build_vocab(train_sen_path)
        vocabulary_len = len(word_to_id)

        train_sentense = [line.strip() for line in fin_train_sen]
        dev_sentense = [line.strip() for line in fin_dev_sen]
        test_sentense = [line.strip() for line in fin_test_sen]

        train_sentence = []
        for sen in train_sentense:
            train_sentence.append(_sentence_to_word_ids(sen, word_to_id))

        dev_sentence = []
        for sen in dev_sentense:
            dev_sentence.append(_sentence_to_word_ids(sen, word_to_id))

        test_sentence = []
        for sen in test_sentense:
            test_sentence.append(_sentence_to_word_ids(sen, word_to_id))

        max_len = max([len(sen) for sen in train_sentence + dev_sentence + test_sentence])

        train_labels = [int(x.strip()) for x in fin_train_label]
        dev_labels = [int(x.strip()) for x in fin_dev_label]
        test_labels = [int(x.strip()) for x in fin_test_label]

        num_classes = max(train_labels) + 1

        train_labels = _trans_label(train_labels, num_classes)
        dev_labels = _trans_label(dev_labels, num_classes)
        test_labels = _trans_label(test_labels, num_classes)


        return train_sentence, dev_sentence, test_sentence, \
               train_labels, dev_labels, test_labels, vocabulary_len