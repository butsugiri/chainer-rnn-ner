# -*- coding: utf-8 -*-
import sys
import chainer
import json
import chainer.links as L
import chainer.serializers as S
import chainer.functions as F
import numpy as np
from chainer import cuda
from .Model import NERTagger
from .DataProcessor import DataProcessor
from datetime import datetime


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

        count = 0
        if self.compute_accuracy:
            for yi, ti in zip(self.y, t):
                if self.accuracy is not None:
                    self.accuracy += self.accfun(yi, ti) * len(ti)
                    count += len(ti)
                else:
                    self.accuracy = self.accfun(yi, ti) * len(ti)
                    count += len(ti)
        return self.loss, self.accuracy, count

class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.epoch = config["total_epoch"]
        self.use_gpu = config["use_gpu"]

        self.data_processor = DataProcessor(
            data_path=config["input"], use_gpu=self.use_gpu)
        self.data_processor.prepare(
            test=config["test_mode"])  # only 100 instances

        self.result_storage = {}
        self.result_storage["result"] = {
            "time_taken": [], "perplexity": []}
        self.result_storage["hyper_params"] = config

        tagger = NERTagger(n_vocab=len(self.data_processor.vocab), embed_dim=config["embed_dim"],
                           hidden_dim=config["hidden_dim"], n_tag=len(self.data_processor.tag))
        self.model = Classifier(tagger)
        if self.use_gpu >= 0:
            chainer.cuda.get_device(self.use_gpu).use()  # make the GPU current
            self.model.to_gpu()

        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)

        if self.use_gpu >= 0:
            self.xp = cuda.cupy
        else:
            self.xp = np

    def run(self):
        for epoch in range(self.epoch):
            sum_loss = 0
            sum_acc = 0
            sum_count = 0
            sys.stderr.write("Currently @ Epoch:{epoch}\n".format(epoch=epoch))
            self.epoch_start_time = datetime.now()
            for xs, ts in self.data_processor.batch_iter():
                # print(xs)
                # print(ts)
                hx = chainer.Variable(
                    self.xp.zeros((1, len(xs), self.config["hidden_dim"]), dtype=self.xp.float32))
                cx = chainer.Variable(
                    self.xp.zeros((1, len(xs), self.config["hidden_dim"]), dtype=self.xp.float32))
                loss, acc, count = self.model(xs, hx, cx, ts)
                sum_loss += loss.data
                sum_acc += acc.data
                sum_count += count
                self.optimizer.target.cleargrads()
                loss.backward()
                self.optimizer.update()

            print("{}\t{}\t{}".format(sum_acc/sum_count, sum_loss, sum_count))
            if (epoch + 1) % 10 == 0:
                self._evaluate(epoch)


    def _evaluate(self, epoch):
        for xs, ts in self.data_processor.batch_iter(train=True):
            hx = chainer.Variable(
                self.xp.zeros((1, len(xs), self.config["hidden_dim"]), dtype=self.xp.float32))
            cx = chainer.Variable(
                self.xp.zeros((1, len(xs), self.config["hidden_dim"]), dtype=self.xp.float32))
            ys = self.model.predictor(xs, hx, cx, train=False)
            for yi, xi, ti in zip(ys, xs, ts):
                pred = F.argmax(F.softmax(yi), axis=1).data
                pred_tags = [self.data_processor.id2tag[i] for i in pred]
                print(" ".join(pred_tags))

                targets = [self.data_processor.id2tag[i] for i in ti]
                print(" ".join(targets))

                # terms = [self.data_processor.id2vocab[i] for i in xi]
                # print(" ".join(terms))

    def _save_stats(self):
        """
        stats.json (accuracy/average loss/total lossを保存するjson) を
        epochごとに書き換えるメソッド
        """
        with open(self.config["log_path"], "w") as result:
            result.write(json.dumps(self.result_storage))
            result.flush()

    def _take_snapshot(self, epoch):
        """
        訓練したモデルをepochごとに保存していくメソッド
        """
        path = "../../result/{time}/model_files/{epoch:02d}.npz".format(
            time=self.config["time"],
            epoch=epoch
        )
        S.save_npz(path, self.model)
