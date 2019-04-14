# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:56:19 2019

@author: YeNz
"""

from __future__ import print_function, division
from builtins import range


import numpy as np
import string
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
# from util import find_analogies

from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances


def get_negative_sampling_distribution(sentences, vocab_size):
  # Pn(w) = prob of word occuring
  # we would like to sample the negative samples
  # such that words that occur more often
  # should be sampled more often

  word_freq = np.zeros(vocab_size)
  word_count = sum(len(sentence) for sentence in sentences)
  for sentence in sentences:
      for word in sentence:
          word_freq[word] += 1

  # smooth it
  p_neg = word_freq**0.75

  # normalize it
  p_neg = p_neg / p_neg.sum()

  assert(np.all(p_neg > 0))
  return p_neg


def get_context(pos, sentence, window_size):
  # input:
  # a sentence of the form: x x x x c c c pos c c c x x x x
  # output:
  # the context word indices: c c c c c c

  start = max(0, pos - window_size)
  end_  = min(len(sentence), pos + window_size)

  context = []
  for ctx_pos, ctx_word_idx in enumerate(sentence[start:end_], start=start):
    if ctx_pos != pos:
      # don't include the input word itself as a target
      context.append(ctx_word_idx)
  return context


def sgd(input_, targets, label, learning_rate, W, V):
  # W[input_] shape: D
  # V[:,targets] shape: D x N
  # activation shape: N
  # print("input_:", input_, "targets:", targets)
  activation = W[input_].dot(V[:,targets])
  prob = sigmoid(activation)

  # gradients
  gV = np.outer(W[input_], prob - label) # D x N
  gW = np.sum((prob - label)*V[:,targets], axis=1) # D

  V[:,targets] -= learning_rate*gV # D x N
  W[input_] -= learning_rate*gW # D

  # return cost (binary cross entropy)
  cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
  return cost.sum()


















