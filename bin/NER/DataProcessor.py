# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
from chainer import cuda
from collections import defaultdict


class DataProcessor(object):

    def __init__(self, data_path, use_gpu):
        self.train_data_path = data_path + "train.clean"
        self.dev_data_path = data_path + "dev.clean"
        self.test_data_path = data_path + "test.clean"
        self.vocab_path = data_path + "vocab.txt"
        self.tag_path = data_path + "ner_tags.txt"

        if use_gpu >= 0:
            self.xp = cuda.cupy
        else:
            self.xp = np

    def prepare(self, test=False):
        # load vocabulary
        sys.stderr.write("loading vocabulary...")
        with open(self.vocab_path, "r") as fi:
            self.vocab = {x.split()[0]: int(x.split()[1]) for x in fi}
            self.id2vocab = {v: k for k,v in self.vocab.items()}
        sys.stderr.write("done.\n")

        sys.stderr.write("loading tags...")
        with open(self.tag_path, "r") as fi:
            self.tag = {x.split()[0]: int(x.split()[1]) for x in fi}
            self.id2tag = {v: k for k,v in self.tag.items()}
        sys.stderr.write("done.\n")

        # load train/dev/test data
        sys.stderr.write("loading dataset...")
        self.train_data = self._load_dataset(self.train_data_path)
        self.dev_data = self._load_dataset(self.dev_data_path)
        self.test_data = self._load_dataset(self.test_data_path)
        if test:
            self.train_data = self.train_data[:100]
            self.dev_data = self.dev_data[:100]
        sys.stderr.write("done.\n")

    def _load_dataset(self, path):
        dataset = []
        with open(path, "r") as input_data:
            for line in input_data:
                tokens = [x for x in json.loads(line)]
                token_ids = [self.vocab[x["surface"]] if x["surface"] in self.vocab else self.vocab[
                    x["pos"] + "<UNK>"] for x in tokens]
                targets = [self.tag[x["target"]] for x in tokens]
                dataset.append((token_ids, targets))
        return dataset
