# -*- coding: utf-8 -*-
"""
訓練済みのモデルとDev or Testデータを入力として与えると
NERタグの予測を出力するスクリプト
"""
import chainer
from chainer import training
from chainer import serializers
from train_model import Classifier, MyUpdater
from NER import NERTagger
from NER import DataProcessor


def main():
    data = DataProcessor(data_path="../work/", use_gpu=-1)
    data.prepare()

    model = Classifier(NERTagger(
        n_vocab=len(data.vocab),
        embed_dim=100,
        hidden_dim=100,
        n_tag=len(data.tag)
    ))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    train = data.train_data
    dev = data.dev_data

    train_iter = chainer.iterators.SerialIterator(train, batch_size=10)
    for batch in train_iter:
        print(batch)

if __name__ == '__main__':
    main()
