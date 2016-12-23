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
from train_model import Classifier
from NER import NERTagger, BiNERTagger, BiCharNERTagger
from NER import DataProcessor

def predict(iter, model_type, model, unit):
    if model_type == "lstm" or model_type == "bilstm":
        for batch in iter:
            xs = [xp.array(x[0], dtype=xp.int32) for x in batch]
            ts = [xp.array(x[2], dtype=xp.int32) for x in batch]
            hx = chainer.Variable(
                xp.zeros((1, len(xs), unit), dtype=xp.float32))
            cx = chainer.Variable(
                xp.zeros((1, len(xs), unit), dtype=xp.float32))
            ys = model.predictor(xs, hx, cx, train=False)
            yield ys, ts
    else:
        for batch in iter:
            xs = [xp.array(x[0], dtype=xp.int32) for x in batch]
            ts = [xp.array(x[2], dtype=xp.int32) for x in batch]
            xxs = [[xp.array(x, dtype=xp.int32) for x in sample[1]] for sample in batch]
            hx = chainer.Variable(
                xp.zeros((1, len(xs), unit+50), dtype=xp.float32))
            cx = chainer.Variable(
                xp.zeros((1, len(xs), unit+50), dtype=xp.float32))
            ys = model.predictor(xs, hx, cx, xxs, train=False)
            yield ys, ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--glove', type=str, default="",
                        help='path to glove vector')
    parser.add_argument('--model-type', dest='model_type', type=str, required=True,
                        help='bilstm / lstm / charlstm')
    parser.add_argument('--model', type=str, required=True,
                        help='path to model file')
    parser.add_argument('--dev', action='store_true')
    parser.set_defaults(dev=False)
    args = parser.parse_args()

    data = DataProcessor(data_path="../work/", use_gpu=-1, test=False)
    data.prepare()

    if args.dev:
        test = data.dev_data
    else:
        test = data.test_data

    if args.model_type == "lstm":
        model = Classifier(NERTagger(
            n_vocab=len(data.vocab),
            embed_dim=100,
            hidden_dim=args.unit,
            n_tag=len(data.tag),
            dropout=None
        ))
    elif args.model_type == 'bilstm':
        model = Classifier(BiNERTagger(
            n_vocab=len(data.vocab),
            embed_dim=100,
            hidden_dim=args.unit,
            n_tag=len(data.tag),
            dropout=None
        ))
    elif args.model_type == 'charlstm':
        model = Classifier(BiCharNERTagger(
            n_vocab=len(data.vocab),
            n_char=len(data.char),
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

    serializers.load_npz(args.model, model)

    test_iter = chainer.iterators.SerialIterator(test, repeat=False, batch_size=10)

    id2tag = data.id2tag
    id2vocab = data.id2vocab

    for ys, ts in predict(test_iter, args.model_type, model, args.unit):
        for y, t in zip(ys, ts):
            pred_ids = F.argmax(F.softmax(y), axis=1).data
            pred_labels = [id2tag[x] for x in pred_ids]

            target_labels = [id2tag[x] for x in t]

            for p, t in zip(pred_labels, target_labels):
                print("{}\t{}".format(p, t))

if __name__ == '__main__':
    main()
