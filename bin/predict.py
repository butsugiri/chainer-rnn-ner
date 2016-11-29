# -*- coding: utf-8 -*-
"""
訓練済みのモデルとDev or Testデータを入力として与えると
NERタグの予測を出力するスクリプト
"""
import chainer
import numpy as xp
import chainer.functions as F
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

    test = data.test_data
    serializers.load_npz("./result/model_iter_4213", model)

    test_iter = chainer.iterators.SerialIterator(test, repeat=False, batch_size=10)

    id2tag = data.id2tag
    id2vocab = data.id2vocab
    for batch in test_iter:
        xs = [xp.array(x[0], dtype=xp.int32) for x in batch]
        ts = [xp.array(x[1], dtype=xp.int32) for x in batch]
        hx = chainer.Variable(
            xp.zeros((1, len(xs), 100), dtype=xp.float32))
        cx = chainer.Variable(
            xp.zeros((1, len(xs), 100), dtype=xp.float32))
        ys = model.predictor(xs, hx, cx, train=False)

        for y, t in zip(ys, ts):
            pred_ids = F.argmax(F.softmax(y), axis=1).data
            pred_labels = [id2tag[x] for x in pred_ids]

            target_labels = [id2tag[x] for x in t]

            for predict, target in zip(pred_labels, target_labels):
                print("{}\t{}".format(predict, target))



if __name__ == '__main__':
    main()
