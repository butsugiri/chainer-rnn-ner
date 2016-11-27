# -*- coding: utf-8 -*-
import chainer
import chainer.links as L
from chainer.links import NStepLSTM


class NERTagger(chainer.Chain):
    """docstring for NERTagger."""

    def __init__(self, n_vocab, n_tag, embed_dim, hidden_dim):
        super(NERTagger, self).__init__(
            embed=L.EmbedID(n_vocab, embed_dim, ignore_label=-1),
            l1=L.NStepLSTM(1, embed_dim, embed_dim, dropout=0.3, use_cudnn=True),
            l2=L.Linear(embed_dim, n_tag),
        )

    def __call__(self, xs, hx, cx, train=True):
        xs = [self.embed(item) for item in xs]
        hy, cy, ys = self.l1(hx, cx, xs, train=train)
        y = [self.l2(item) for item in ys]
        return y
