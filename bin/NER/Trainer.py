# -*- coding: utf-8 -*-
import sys
import chainer
import json
import chainer.links as L
import chainer.serializers as S
import numpy as np
from chainer import cuda
from .Model import NERTagger
from .DataProcessor import DataProcessor
from datetime import datetime

class Classifier(chainer.link.Chain):
    pass


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

        # self.model = L.Classifier(
        #     NERTagger(vocab=self.data_processor.vocab, embed_dim=config["embed_dim"]))
        self.model = NERTagger(n_vocab=len(self.data_processor.vocab), embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"], n_tag=len(self.data_processor.tag))
        self.model.compute_accuracy = False
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
            sys.stderr.write("Currently @ Epoch:{epoch}\n".format(epoch=epoch))
            self.epoch_start_time = datetime.now()
            for xs, ts in self.data_processor.batch_iter():
                # print(xs)
                # print(ts)
                hx = chainer.Variable(
                self.xp.zeros((1, len(xs), self.config["hidden_dim"]), dtype=self.xp.float32))
                cx = chainer.Variable(
                self.xp.zeros((1, len(xs), self.config["hidden_dim"]), dtype=self.xp.float32))
                y = self.model(xs, hx, cx)

                exit()






                for curr_words, next_words in zip(batch.T, batch[:, 1:].T):
                    accum_loss += self.model(curr_words, next_words)
                self.optimizer.target.cleargrads()
                accum_loss.backward()
                self.optimizer.update()
                self.model.predictor.reset_state()
            self.epoch_end_time = datetime.now()

            if epoch + 1 % 50 == 0:
                self._take_snapshot(epoch)
                self._evaluate(epoch)

    def _evaluate(self, epoch):
        accum_loss = 0
        self.model.predictor.reset_state()

        dataset_count = 0
        for batch in self.data_processor.batch_iter(train=self.config["cheat"], batch_size=self.config["batch_size"], shuffle_batch=False):
            for curr_words, next_words in zip(batch.T, batch[:, 1:].T):
                accum_loss += cuda.to_cpu(self.model(curr_words,
                                                     next_words).data)
                dataset_count += 1
            self.model.predictor.reset_state()

        perp = np.exp(float(accum_loss) / (dataset_count * self.config["batch_size"]))

        # epoch単位の実行時間の計算
        td = self.epoch_end_time - self.epoch_start_time  # td: timedelta
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_taken = "{hours:02d}:{minutes:02d}".format(
            hours=hours, minutes=minutes)

        # epoch単位の各種statsの出力
        sys.stderr.write("Epoch: #{epoch}\tTotal Loss:{total:.4f}\tTime Taken:{time_taken}\tPerplexity:{perp}\n".format(
            epoch=epoch,
            total=accum_loss,
            time_taken=time_taken,
            perp=perp
        ))

        self.result_storage["result"]["time_taken"].append(time_taken)
        self.result_storage["result"]["perplexity"].append(perp)
        self._save_stats()

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
