# -*- coding: utf-8 -*-
import sys
import numpy as np
from chainer import cuda
from collections import defaultdict


class DataProcessor(object):

    def __init__(self, data_path, use_gpu):
        self.train_data_path = data_path + "train.txt"
        self.dev_data_path = data_path + "dev.txt"
        self.test_data_path = data_path + "test.txt"
        self.vocab_path = data_path + "vocab.txt"

        if use_gpu >= 0:
            self.xp = cuda.cupy
        else:
            self.xp = np

    def prepare(self, test=False):
        # load vocabulary
        sys.stderr.write("loading vocabulary...")
        with open(self.vocab_path, "r") as fi:
            self.vocab = {unicode(x.split()[0]): int(x.split()[1]) for x in fi}
        sys.stderr.write("done.\n")

        # load train/dev/test data
        sys.stderr.write("loading dataset...")
        self.train_data = self._load_dataset(self.train_data_path)
        self.dev_data = self._load_dataset(self.dev_data_path)
        if test:
            self.train_data = self.train_data[:100]
            self.dev_data = self.dev_data[:100]
        sys.stderr.write("done.\n")

    def _load_dataset(self, path):
        dataset = []
        with open(path, "r") as input_data:
            for line in input_data:
                tokens = [unicode(x) for x in line.rstrip().split(" ")]
                token_ids = [self.vocab[x] if x in self.vocab else self.vocab[
                    u"<UNK>"] for x in tokens]
                dataset.append(token_ids)
        return dataset

    def batch_iter(self, batch_size=16, shuffle_batch=True, train=True):
        if train:
            dataset = self.train_data
        else:
            dataset = self.dev_data

        same_length = defaultdict(list)
        for tokens in dataset:
            same_length[len(tokens)].append(tokens)

        for tokens in same_length.itervalues():
            N = len(tokens)
            if shuffle_batch:
                np.random.shuffle(tokens)
            for i in xrange(0, N, batch_size):
                yield self.xp.array(tokens[i:i + batch_size], dtype=np.int32)
