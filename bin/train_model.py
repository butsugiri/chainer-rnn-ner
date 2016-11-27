# -*- coding: utf-8 -*-
import copy
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import reporter
from chainer.training import extensions

from NER.Trainer import Classifier
from NER import Resource
from NER import NERTagger
from NER import DataProcessor
import numpy as xp


class MyUpdater(training.StandardUpdater):

    def __init__(self, iterator, optimizer):
        super(MyUpdater, self).__init__(iterator=iterator, optimizer=optimizer)

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
    data_processor = DataProcessor(data_path="../work/", use_gpu=-1)
    data_processor.prepare(test=True)

    model = Classifier(NERTagger(
        n_vocab=len(data_processor.vocab),
        embed_dim=10,
        hidden_dim=10,
        n_tag=len(data_processor.tag)
    ))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = data_processor.train_data
    dev = data_processor.dev_data

    train_iter = chainer.iterators.SerialIterator(train, batch_size=10)
    dev_iter = chainer.iterators.SerialIterator(dev, batch_size=10, repeat=False)

    updater = MyUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (10, 'epoch'), out="result")

    trainer.extend(MyEvaluator(dev_iter, optimizer.target, device=-1))
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.run()

if __name__ == "__main__":
    main()
