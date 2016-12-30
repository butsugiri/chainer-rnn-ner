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
from NER import CRFNERTagger, CRFBiNERTagger, CRFBiCharNERTagger
from NER import DataProcessor
from tqdm import tqdm

def predict(iter, model_type, model, unit):
    if model_type == "lstm" or model_type == "bilstm":
        for batch in iter:
            inds = xp.argsort([-len(x[0]) for x in batch]).astype('i')
            xs = [xp.array(batch[i][0], dtype=xp.int32) for i in inds]
            ts = [xp.array(batch[i][2], dtype=xp.int32) for i in inds]

            hx = chainer.Variable(
                xp.zeros((1, len(xs), unit), dtype=xp.float32))
            cx = chainer.Variable(
                xp.zeros((1, len(xs), unit), dtype=xp.float32))
            ys, ts = model.predict(xs, hx, cx, ts, train=False)
            yield ys, ts
    else:
        for batch in iter:
            inds = xp.argsort([-len(x[0]) for x in batch]).astype('i')
            xs = [xp.array(batch[i][0], dtype=xp.int32) for i in inds]
            ts = [xp.array(batch[i][2], dtype=xp.int32) for i in inds]
            xxs = [[xp.array(x, dtype=xp.int32) for x in batch[i][1]] for i in inds]

            hx = chainer.Variable(
                xp.zeros((1, len(xs), unit+50), dtype=xp.float32))
            cx = chainer.Variable(
                xp.zeros((1, len(xs), unit+50), dtype=xp.float32))
            ys, ts = model.predict(xs, hx, cx, xxs, ts, train=False)
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
        model = CRFNERTagger(
            n_vocab=len(data.vocab),
            embed_dim=100,
            hidden_dim=args.unit,
            n_tag=len(data.tag),
            dropout=None
        )
    elif args.model_type == 'bilstm':
        model = CRFBiNERTagger(
            n_vocab=len(data.vocab),
            embed_dim=100,
            hidden_dim=args.unit,
            n_tag=len(data.tag),
            dropout=None
        )
    elif args.model_type == 'charlstm':
        model = CRFBiCharNERTagger(
            n_vocab=len(data.vocab),
            n_char=len(data.char),
            embed_dim=100,
            hidden_dim=args.unit,
            n_tag=len(data.tag),
            dropout=None
        )

    # load glove vector
    if args.glove:
        sys.stderr.write("loading glove...")
        model.predictor.load_glove(args.glove, data.vocab)
        sys.stderr.write("done.\n")

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    serializers.load_npz(args.model, model)

    test_iter = chainer.iterators.SerialIterator(test, repeat=False, shuffle=False, batch_size=10)

    id2tag = data.id2tag
    id2vocab = data.id2vocab

    for ys, ts in tqdm(predict(test_iter, args.model_type, model, args.unit)):
        # minibatch-unit-loop
        ys = [[id2tag[i] for i in y.data] for y in F.transpose_sequence(ys)]
        ts = [[id2tag[i] for i in t.data] for t in F.transpose_sequence(ts)]
        # instance-loop
        for predict_seq, target_seq in zip(ys, ts):
            for p, t in zip(predict_seq, target_seq):
                print("{}\t{}".format(p, t))
            print()

if __name__ == '__main__':
    main()
