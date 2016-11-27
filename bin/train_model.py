# -*- coding: utf-8 -*-
import copy
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import reporter
from chainer import cuda
from chainer.training import extensions

from NER import Resource
from NER import NERTagger
from NER import DataProcessor
import numpy as xp


class Classifier(chainer.link.Chain):
    compute_accuracy = True

    def __init__(self, predictor, lossfun=F.softmax_cross_entropy,
                 accfun=F.accuracy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, *args, train=True):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x, train)
        for yi, ti, in zip(self.y, t):
            if self.loss is not None:
                self.loss += self.lossfun(yi, ti)
            else:
                self.loss = self.lossfun(yi, ti)
        reporter.report({'loss': self.loss}, self)

        count = 0
        if self.compute_accuracy:
            for yi, ti in zip(self.y, t):
                if self.accuracy is not None:
                    self.accuracy += self.accfun(yi, ti) * len(ti)
                    count += len(ti)
                else:
                    self.accuracy = self.accfun(yi, ti) * len(ti)
                    count += len(ti)
            reporter.report({'accuracy': self.accuracy / count}, self)
        return self.loss, self.accuracy, count


class MyUpdater(training.StandardUpdater):

    def __init__(self, iterator, optimizer, device):
        super(MyUpdater, self).__init__(iterator=iterator, optimizer=optimizer)
        if device >= 0:
            xp = cuda.cupy

    def update_core(self):
        batch = self._iterators['main'].next()
        optimizer = self._optimizers['main']
        xs = [xp.array(x[0], dtype=xp.int32) for x in batch]
        ts = [xp.array(x[1], dtype=xp.int32) for x in batch]

        optimizer.target.cleargrads()
        hx = chainer.Variable(
            xp.zeros((1, len(xs), 10), dtype=xp.float32))
        cx = chainer.Variable(
            xp.zeros((1, len(xs), 10), dtype=xp.float32))
        loss, accuracy, count = optimizer.target(xs, hx, cx, ts, train=True)
        loss.backward()
        optimizer.update()

class MyEvaluator(extensions.Evaluator):
    def __init__(self, iterator, target, device):
        super(MyEvaluator, self).__init__(iterator=iterator, target=target, device=device)
        if device >= 0:
            xp = cuda.cupy

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        it = copy.copy(iterator) # これがないと1回しかEvaluationが走らない
        summary = reporter.DictSummary()
        i = 0
        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                xs = [xp.array(x[0], dtype=xp.int32) for x in batch]
                ts = [xp.array(x[1], dtype=xp.int32) for x in batch]
                hx = chainer.Variable(
                    xp.zeros((1, len(xs), 10), dtype=xp.float32))
                cx = chainer.Variable(
                    xp.zeros((1, len(xs), 10), dtype=xp.float32))
                loss = target(xs, hx, cx, ts, train=False)

            summary.add(observation)
        return summary.compute_mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()
    data_processor = DataProcessor(data_path="../work/", use_gpu=args.gpu)
    data_processor.prepare()

    model = Classifier(NERTagger(
        n_vocab=len(data_processor.vocab),
        embed_dim=args.unit,
        hidden_dim=args.unit,
        n_tag=len(data_processor.tag)
    ))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    train = data_processor.train_data
    dev = data_processor.dev_data

    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize)
    dev_iter = chainer.iterators.SerialIterator(dev, batch_size=args.batchsize, repeat=False)

    updater = MyUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (10, 'epoch'), out="result")

    trainer.extend(MyEvaluator(dev_iter, optimizer.target, device=args.gpu))
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))

    trainer.extend(extensions.ProgressBar(update_interval=100))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.run()

if __name__ == "__main__":
    main()
