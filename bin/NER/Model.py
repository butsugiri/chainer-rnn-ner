# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links import NStepLSTM


class NERTagger(chainer.Chain):
    """docstring for NERTagger."""

    def __init__(self, n_vocab, n_tag, embed_dim, hidden_dim, dropout):
        super(NERTagger, self).__init__(
            embed=L.EmbedID(n_vocab, embed_dim, ignore_label=-1),
            l1=L.NStepLSTM(1, embed_dim, embed_dim, dropout=0.3, use_cudnn=True),
            l2=L.Linear(embed_dim, n_tag),
        )
        if dropout:
            self.dropout = True
        else:
            self.dropout = False

    def __call__(self, xs, hx, cx, train=True):
        xs = [self.embed(item) for item in xs]
        if self.dropout and train:
            xs = [F.dropout(item) for item in xs]
        hy, cy, ys = self.l1(hx, cx, xs, train=False) #don't use dropout
        y = [self.l2(item) for item in ys]
        return y

    def load_glove(self, path, vocab):
        with open(path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=np.float32)
                    self.embed.W.data[vocab[word]] = vec


class BiNERTagger(chainer.Chain):
    """docstring for BiNERTagger."""

    def __init__(self, n_vocab, n_tag, embed_dim, hidden_dim, dropout):
        super(BiNERTagger, self).__init__(
            embed=L.EmbedID(n_vocab, embed_dim, ignore_label=-1),
            forward_l1=L.NStepLSTM(1, embed_dim, embed_dim, dropout=0.3, use_cudnn=True),
            backward_l1=L.NStepLSTM(1, embed_dim, embed_dim, dropout=0.3, use_cudnn=True),
            l2=L.Linear(embed_dim * 2, n_tag),
        )
        if dropout:
            self.dropout = True
        else:
            self.dropout = False


    def __call__(self, xs, hx, cx, train=True):
        xs = [self.embed(item) for item in xs]
        if self.dropout and train:
            xs = [F.dropout(item) for item in xs]
        forward_hy, forward_cy, forward_ys = self.forward_l1(hx, cx, xs, train=False) #don't use dropout
        backward_hy, backward_cy, backward_ys = self.backward_l1(hx, cx, xs, train=False) #don't use dropout
        ys = [F.concat([forward, backward], axis=1) for forward, backward in zip(forward_ys, backward_ys)]
        y = [self.l2(item) for item in ys]
        return y


    def load_glove(self, path, vocab):
        with open(path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=np.float32)
                    self.embed.W.data[vocab[word]] = vec
