# -*- coding: utf-8 -*-
import chainer
import chainer.links as L


class RNNLM(chainer.Chain):
    """docstring for RNNLM."""

    def __init__(self, vocab, embed_dim):
        n_vocab = len(vocab)
        super(RNNLM, self).__init__(
            embed=L.EmbedID(n_vocab, embed_dim, ignore_label=-1),
            l1=L.LSTM(embed_dim, embed_dim),
            l2=L.Linear(embed_dim, n_vocab),
        )
        self.l1.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        y = self.l2(h1)
        return y

    def reset_state(self):
        self.l1.reset_state()

    def set_state(self, c, h):
        self.l1.set_state(c, h)

    def get_state(self):
        return (self.l1.c, self.l1.h)
