# -*- coding: utf-8 -*-
import sys
import os
import json
import copy
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import reporter
from chainer import cuda
from chainer.training import extensions
from datetime import datetime

from NER import Resource
from NER import CRFNERTagger, CRFBiNERTagger, CRFBiCharNERTagger
from NER import DataProcessor
import numpy as xp


class LSTMUpdater(training.StandardUpdater):

    def __init__(self, iterator, optimizer, device, unit):
        super(LSTMUpdater, self).__init__(iterator=iterator, optimizer=optimizer)
        if device >= 0:
            self.xp = cuda.cupy
        else:
            self.xp = xp
        self.unit = unit

    def update_core(self):
        batch = self._iterators['main'].next()
        optimizer = self._optimizers['main']
        inds = xp.argsort([-len(x[0]) for x in batch]).astype('i')
        xs = [self.xp.array(batch[i][0], dtype=self.xp.int32) for i in inds]
        ts = [self.xp.array(batch[i][2], dtype=self.xp.int32) for i in inds]

        optimizer.target.cleargrads()
        hx = chainer.Variable(
            self.xp.zeros((1, len(xs), self.unit), dtype=self.xp.float32))
        cx = chainer.Variable(
            self.xp.zeros((1, len(xs), self.unit), dtype=self.xp.float32))
        loss, accuracy, count = optimizer.target(
            xs, hx, cx, ts, train=True)
        loss.backward()
        optimizer.update()


class CharLSTMUpdater(training.StandardUpdater):

    def __init__(self, iterator, optimizer, device, unit):
        super(CharLSTMUpdater, self).__init__(iterator=iterator, optimizer=optimizer)
        if device >= 0:
            self.xp = cuda.cupy
        else:
            self.xp = xp
        self.unit = unit

    def update_core(self):
        batch = self._iterators['main'].next()
        optimizer = self._optimizers['main']
        inds = xp.argsort([-len(x[0]) for x in batch]).astype('i')
        xs = [self.xp.array(batch[i][0], dtype=self.xp.int32) for i in inds]
        ts = [self.xp.array(batch[i][2], dtype=self.xp.int32) for i in inds]
        xxs = [[self.xp.array(x, dtype=self.xp.int32) for x in batch[i][1]] for i in inds]

        optimizer.target.cleargrads()
        hx = chainer.Variable(
            self.xp.zeros((1, len(xs), self.unit + 50), dtype=self.xp.float32))
        cx = chainer.Variable(
            self.xp.zeros((1, len(xs), self.unit + 50), dtype=self.xp.float32))
        loss, accuracy, count = optimizer.target(
            xs, hx, cx, xxs, ts, train=True)
        loss.backward()
        optimizer.update()


class LSTMEvaluator(extensions.Evaluator):

    def __init__(self, iterator, target, device, unit):
        super(LSTMEvaluator, self).__init__(
            iterator=iterator, target=target, device=device)
        if device >= 0:
            self.xp = cuda.cupy
        else:
            self.xp = xp
        self.unit = unit

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        it = copy.copy(iterator)  # これがないと1回しかEvaluationが走らない
        summary = reporter.DictSummary()
        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                inds = xp.argsort([-len(x[0]) for x in batch]).astype('i')
                xs = [self.xp.array(batch[i][0], dtype=self.xp.int32) for i in inds]
                ts = [self.xp.array(batch[i][2], dtype=self.xp.int32) for i in inds]
                hx = chainer.Variable(
                    self.xp.zeros((1, len(xs), self.unit), dtype=self.xp.float32))
                cx = chainer.Variable(
                    self.xp.zeros((1, len(xs), self.unit), dtype=self.xp.float32))

                loss = target(xs, hx, cx, ts, train=False)

            summary.add(observation)
        return summary.compute_mean()


class CharLSTMEvaluator(extensions.Evaluator):

    def __init__(self, iterator, target, device, unit):
        super(CharLSTMEvaluator, self).__init__(
            iterator=iterator, target=target, device=device)
        if device >= 0:
            self.xp = cuda.cupy
        else:
            self.xp = xp
        self.unit = unit

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        it = copy.copy(iterator)  # これがないと1回しかEvaluationが走らない
        summary = reporter.DictSummary()
        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                inds = xp.argsort([-len(x[0]) for x in batch]).astype('i')
                xs = [self.xp.array(batch[i][0], dtype=self.xp.int32) for i in inds]
                ts = [self.xp.array(batch[i][2], dtype=self.xp.int32) for i in inds]
                xxs = [[self.xp.array(x, dtype=self.xp.int32) for x in batch[i][1]] for i in inds]

                hx = chainer.Variable(
                    self.xp.zeros((1, len(xs), self.unit + 50), dtype=self.xp.float32))
                cx = chainer.Variable(
                    self.xp.zeros((1, len(xs), self.unit + 50), dtype=self.xp.float32))

                loss = target(xs, hx, cx, xxs, ts, train=False)

            summary.add(observation)
        return summary.compute_mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=5,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=6,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--glove', type=str, default="",
                        help='path to glove vector')
    parser.add_argument('--dropout', action='store_true',
                        help='use dropout?')
    parser.set_defaults(dropout=False)
    parser.add_argument('--model-type', dest='model_type', type=str, required=True,
                        help='bilstm / lstm / char-bi-lstm')
    parser.add_argument('--final-layer', default='withCRF', type=str)
    args = parser.parse_args()

    # save configurations to file
    start_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    dest = "../result/" + start_time
    os.makedirs(dest)
    with open(os.path.join(dest, "settings.json"), "w") as fo:
        fo.write(json.dumps(vars(args), sort_keys=True, indent=4))

    # 学習/validation データの準備
    data_processor = DataProcessor(
        data_path="../work/", use_gpu=args.gpu, test=args.test)
    data_processor.prepare()
    train = data_processor.train_data
    dev = data_processor.dev_data

    train_iter = chainer.iterators.SerialIterator(
        train, batch_size=args.batchsize)
    dev_iter = chainer.iterators.SerialIterator(
        dev, batch_size=args.batchsize, repeat=False)

    # モデルの準備
    optimizer = chainer.optimizers.Adam()
    if args.model_type == "bilstm":
        sys.stderr.write("Using Bidirectional LSTM\n")
        model = CRFBiNERTagger(
            n_vocab=len(data_processor.vocab),
            embed_dim=args.unit,
            hidden_dim=args.unit,
            n_tag=len(data_processor.tag),
            dropout=args.dropout
        )
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(5))
        updater = LSTMUpdater(train_iter, optimizer,
                            device=args.gpu, unit=args.unit)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                                   out="../result/" + start_time)
        trainer.extend(LSTMEvaluator(dev_iter, optimizer.target,
                                   device=args.gpu, unit=args.unit))

    elif args.model_type == "lstm":
        sys.stderr.write("Using Normal LSTM\n")
        model = CRFNERTagger(
            n_vocab=len(data_processor.vocab),
            embed_dim=args.unit,
            hidden_dim=args.unit,
            n_tag=len(data_processor.tag),
            dropout=args.dropout
        )
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(5))
        updater = LSTMUpdater(train_iter, optimizer,
                            device=args.gpu, unit=args.unit)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                                   out="../result/" + start_time)
        trainer.extend(LSTMEvaluator(dev_iter, optimizer.target,
                                   device=args.gpu, unit=args.unit))

    elif args.model_type == "charlstm":
        sys.stderr.write("Using Bidirectional LSTM with character encoding\n")
        model = CRFBiCharNERTagger(
            n_vocab=len(data_processor.vocab),
            n_char=len(data_processor.char),
            embed_dim=args.unit,
            hidden_dim=args.unit,
            n_tag=len(data_processor.tag),
            dropout=args.dropout
        )
        optimizer.setup(model)
        updater = CharLSTMUpdater(train_iter, optimizer,
                            device=args.gpu, unit=args.unit)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                                   out="../result/" + start_time)
        trainer.extend(CharLSTMEvaluator(dev_iter, optimizer.target,
                                    device=args.gpu, unit=args.unit))

    # 必要とあらばGPUを使う
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # load glove vector
    if args.glove:
        sys.stderr.write("loading glove...")
        model.load_glove(args.glove, data_processor.vocab)
        sys.stderr.write("done.\n")

    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}', trigger=chainer.training.triggers.MaxValueTrigger('validation/main/accuracy')))

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.run()

if __name__ == "__main__":
    main()
