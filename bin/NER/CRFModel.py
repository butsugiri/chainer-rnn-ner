# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links import NStepLSTM
from chainer import reporter


class CRFTaggerBase(chainer.Chain):

    def __init__(self):
        pass

    # このメソッドは単方向・双方向LSTMで使いたい
    def load_glove(self, path, vocab):
        with open(path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=np.float32)
                    self.embed.W.data[vocab[word]] = vec


class CRFNERTagger(CRFTaggerBase):
    """single LSTM"""

    def __init__(self, n_vocab, n_tag, embed_dim, hidden_dim, dropout):
        super(CRFTaggerBase, self).__init__(
            embed=L.EmbedID(n_vocab, embed_dim, ignore_label=-1),
            l1=L.NStepLSTM(1, embed_dim, embed_dim, dropout=0, use_cudnn=True),
            l2=L.Linear(embed_dim, n_tag),
            crf=L.CRF1d(n_tag)
        )
        if dropout:
            self.dropout = True
        else:
            self.dropout = False

    def __call__(self, xs, hx, cx, ts, train=True):
        ys, ts = self.encode_sequence(xs, hx, cx, ts, train)
        loss = self.crf(ys, ts)
        reporter.report({'loss': loss}, self)

        _, predicts = self.crf.argmax(ys)
        count = 0
        correct = 0
        for t, predict in zip(ts, predicts):
            correct += self.xp.sum(t.data == predict)
            count += len(t.data)
        accuracy = correct * 1.0 / count
        reporter.report({'accuracy': accuracy}, self)
        return loss, accuracy, count

    def predict(self, xs, hx, cx, ts, train=False):
        ys, ts = self.encode_sequence(xs, hx, cx, ts, train)
        _, predicts = self.crf.argmax(ys)
        return predicts, ts

    def encode_sequence(self, xs, hx, cx, ts, train):
        xs = [self.embed(item) for item in xs]
        if self.dropout and train:
            xs = [F.dropout(item) for item in xs]
        hy, cy, ys = self.l1(hx, cx, xs, train=train)  # don't use dropout
        ys = [self.l2(item) for item in ys]
        ys = F.transpose_sequence(ys)
        ts = F.transpose_sequence(ts)
        return ys, ts


class CRFBiNERTagger(CRFTaggerBase):
    """bi-directional LSTM"""

    def __init__(self, n_vocab, n_tag, embed_dim, hidden_dim, dropout):
        super(CRFTaggerBase, self).__init__(
            embed=L.EmbedID(n_vocab, embed_dim, ignore_label=-1),
            forward_l1=L.NStepLSTM(
                1, embed_dim, embed_dim, dropout=0, use_cudnn=True),
            backward_l1=L.NStepLSTM(
                1, embed_dim, embed_dim, dropout=0, use_cudnn=True),
            l2=L.Linear(embed_dim * 2, n_tag),
            crf=L.CRF1d(n_tag)
        )
        if dropout:
            self.dropout = True
        else:
            self.dropout = False

    def __call__(self, xs, hx, cx, ts, train=True):
        ys, ts = self.encode_sequence(xs, hx, cx, ts, train)
        loss = self.crf(ys, ts)
        reporter.report({'loss': loss}, self)

        _, predicts = self.crf.argmax(ys)
        count = 0
        correct = 0
        for t, predict in zip(ts, predicts):
            correct += self.xp.sum(t.data == predict)
            count += len(t.data)
        accuracy = correct * 1.0 / count
        reporter.report({'accuracy': accuracy}, self)
        return loss, accuracy, count

    def predict(self, xs, hx, cx, ts, train=False):
        ys, ts = self.encode_sequence(xs, hx, cx, ts, train)
        _, predicts = self.crf.argmax(ys)
        return predicts, ts

    def encode_sequence(self, xs, hx, cx, ts, train):
        xs = [self.embed(item) for item in xs]
        xs_backward = [item[::-1] for item in xs]
        if self.dropout and train:
            xs = [F.dropout(item) for item in xs]
            xs_backward = [F.dropout(item) for item in xs_backward]
        forward_hy, forward_cy, forward_ys = self.forward_l1(
            hx, cx, xs, train=train)
        backward_hy, backward_cy, backward_ys = self.backward_l1(
            hx, cx, xs_backward, train=train)
        ys = [F.concat([forward, backward[::-1]], axis=1)
              for forward, backward in zip(forward_ys, backward_ys)]
        ys = [self.l2(item) for item in ys]
        ys = F.transpose_sequence(ys)
        ts = F.transpose_sequence(ts)
        return ys, ts


class CRFBiCharNERTagger(CRFTaggerBase):
    """bi-directional LSTM with char-based encoding"""

    def __init__(self, n_vocab, n_char, n_tag, embed_dim, hidden_dim, dropout):
        super(CRFTaggerBase, self).__init__(
            embed=L.EmbedID(n_vocab, embed_dim, ignore_label=-1),
            # character embeddingは25で決め打ち
            char_embed=L.EmbedID(n_char, 25, ignore_label=-1),
            forward_l1=L.NStepLSTM(
                1, embed_dim + 50, embed_dim + 50, dropout=0, use_cudnn=True),
            backward_l1=L.NStepLSTM(
                1, embed_dim + 50, embed_dim + 50, dropout=0, use_cudnn=True),
            l2=L.Linear((embed_dim + 50) * 2, n_tag),
            forward_char=L.NStepLSTM(1, 25, 25, dropout=0, use_cudnn=True),
            backward_char=L.NStepLSTM(1, 25, 25, dropout=0, use_cudnn=True),
            crf=L.CRF1d(n_tag)
        )
        if dropout:
            self.dropout = True
        else:
            self.dropout = False

    def __call__(self, xs, hx, cx, xxs, ts, train=True):
        ys, ts = self.encode_sequence(xs, hx, cx, xxs, ts, train)
        loss = self.crf(ys, ts)
        reporter.report({'loss': loss}, self)

        _, predicts = self.crf.argmax(ys)
        count = 0
        correct = 0
        for t, predict in zip(ts, predicts):
            correct += self.xp.sum(t.data == predict)
            count += len(t.data)
        accuracy = correct * 1.0 / count
        reporter.report({'accuracy': accuracy}, self)
        return loss, accuracy, count

    def predict(self, xs, hx, cx, ts, train=False):
        ys, ts = self.encode_sequence(xs, hx, cx, xxs, ts, train)
        _, predicts = self.crf.argmax(ys)
        return predicts, ts

    def encode_sequence(self, xs, hx, cx, xxs, ts, train):
        forward_char_embeds = [
            [self.char_embed(item) for item in items] for items in xxs]
        backward_char_embeds = [[item[::-1] for item in items]
                                for items in forward_char_embeds]

        forward_encodings = []
        backward_encodings = []
        for forward, backward in zip(forward_char_embeds, backward_char_embeds):
            hhx = chainer.Variable(
                self.xp.zeros((1, len(forward), 25), dtype=self.xp.float32))
            ccx = chainer.Variable(
                self.xp.zeros((1, len(forward), 25), dtype=self.xp.float32))
            _, __, forward_char_encs = self.forward_char(hhx, ccx, forward)
            _, __, backward_char_encs = self.backward_char(hhx, ccx, backward)
            forward_encodings.append([x[-1] for x in forward_char_encs])
            backward_encodings.append([x[-1] for x in backward_char_encs])

        forward_encodings = [F.vstack(x) for x in forward_encodings]
        backward_encodings = [F.vstack(x) for x in backward_encodings]
        xs = [self.embed(item) for item in xs]
        xs_forward = [F.concat([x, y, z], axis=1) for x, y, z in zip(
            xs, forward_encodings, backward_encodings)]
        xs_backward = [x[::-1] for x in xs_forward]

        if self.dropout and train:
            xs_forward = [F.dropout(item) for item in xs_forward]
            xs_backward = [F.dropout(item) for item in xs_backward]
        forward_hy, forward_cy, forward_ys = self.forward_l1(
            hx, cx, xs_forward, train=train)  # don't use dropout
        backward_hy, backward_cy, backward_ys = self.backward_l1(
            hx, cx, xs_backward, train=train)  # don't use dropout
        ys = [F.concat([forward, backward[::-1]], axis=1)
              for forward, backward in zip(forward_ys, backward_ys)]
        ys = [self.l2(item) for item in ys]

        ys = F.transpose_sequence(ys)
        ts = F.transpose_sequence(ts)
        return ys, ts
