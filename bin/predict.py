# -*- coding: utf-8 -*-
"""
訓練済みのモデルとDev or Testデータを入力として与えると
NERタグの予測を出力するスクリプト
"""
import sys
import chainer
import argparse
import numpy as xp
import chainer.functions as F
from chainer import training
from chainer import serializers
from train_model import Classifier, MyUpdater
from NER import NERTagger, BiNERTagger, BiCharNERTagger
from NER import DataProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--glove', type=str, default="",
                        help='path to glove vector')
    parser.add_argument('--bilstm', action='store_true',
                        help='use bi-lstm?')
    parser.add_argument('--model', type=str, required=True,
                        help='use bi-lstm?')
    parser.set_defaults(bilstm=False)
    args = parser.parse_args()

    data = DataProcessor(data_path="../work/", use_gpu=-1, test=False)
    data.prepare()

    if args.bilstm:
        model = Classifier(BiCharNERTagger(
            n_vocab=len(data.vocab),
            n_char=len(data.char),
            embed_dim=100,
            hidden_dim=args.unit,
            n_tag=len(data.tag),
            dropout=None
        ))
    else:
        model = Classifier(NERTagger(
            n_vocab=len(data.vocab),
            embed_dim=100,
            hidden_dim=args.unit,
            n_tag=len(data.tag),
            dropout=None
        ))

    # load glove vector
    if args.glove:
        sys.stderr.write("loading glove…")
        model.predictor.load_glove(args.glove, data.vocab)
        sys.stderr.write("done.\n")

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    test = data.test_data
    serializers.load_npz(args.model, model)

    test_iter = chainer.iterators.SerialIterator(test, repeat=False, batch_size=10)

    id2tag = data.id2tag
    id2vocab = data.id2vocab
    for batch in test_iter:
        xs = [xp.array(x[0], dtype=xp.int32) for x in batch]
        ts = [xp.array(x[2], dtype=xp.int32) for x in batch]
        xxs = [[xp.array(x, dtype=xp.int32) for x in sample[1]] for sample in batch]
        hx = chainer.Variable(
            xp.zeros((1, len(xs), args.unit+50), dtype=xp.float32))
        cx = chainer.Variable(
            xp.zeros((1, len(xs), args.unit+50), dtype=xp.float32))
        ys = model.predictor(xs, hx, cx, train=False)

        for y, t in zip(ys, ts):
            pred_ids = F.argmax(F.softmax(y), axis=1).data
            pred_labels = [id2tag[x] for x in pred_ids]

            target_labels = [id2tag[x] for x in t]

            for predict, target in zip(pred_labels, target_labels):
                print("{}\t{}".format(predict, target))



if __name__ == '__main__':
    main()
