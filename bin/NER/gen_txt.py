# -*- coding: utf-8 -*-
import argparse
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from Model import RNNLM
from chainer import cuda, serializers


class SentenceGenerator(object):
    """
    GPUは諦めてる(生成はCPUでもスケールする（僕の用途では,多分)
    generateメソッドから各種生成法に分岐させる．
    """
    # TODO: primetextも確率的(orランダムに)選べるように

    def __init__(self, vocab_path, model_path, n_unit):
        with open(vocab_path) as fi:
            # vocab: wordからindexへ
            self.vocab = {unicode(x.split()[0]): int(x.split()[1]) for x in fi}
            # ivocab: indexからvocabへ
            self.ivocab = {idx: w for w, idx in self.vocab.iteritems()}
        self.model = L.Classifier(RNNLM(vocab=self.vocab, embed_dim=n_unit))
        serializers.load_npz(model_path, self.model)

        self.n_unit = n_unit

    def generate(self, how="probablistic", primetext=u"<BOS>", max_length=100, width=3):
        self.primetext = primetext
        self.max_length = max_length
        if primetext in self.vocab:
            if how == "probablistic":
                out = self._probablistic()
            elif how == "beam_search":
                out = self._beam_search(width)
            elif how == "greedy":
                out = self._beam_search(width=1)
            return out
        else:
            return "ERROR"

    def _probablistic(self):
        prev_word = np.asarray([self.vocab[self.primetext]], dtype=np.int32)

        sentence = []
        if self.primetext != u"<BOS>":
            sentence.append(self.primetext)

        for i in xrange(self.max_length):
            prob = F.softmax(self.model.predictor(prev_word))
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
            if self.ivocab[index] == u'<EOS>':
                sentence.append("。")
                break
            else:
                sentence.append(self.ivocab[index])

            prev_word = chainer.Variable(np.array([index], dtype=np.int32))
        self.model.predictor.reset_state()
        return " ".join(sentence)

    def _beam_search(self, width=3):
        initial_state = {
            "path": [self.primetext],
            "cell": chainer.Variable(np.zeros((1, self.n_unit), dtype=np.float32)),
            "hidden": chainer.Variable(np.zeros((1, self.n_unit), dtype=np.float32)),
            "prob": 0
        }
        candidates = [initial_state]

        length = 0
        while length < self.max_length:
            temp = [x for x in candidates if x["path"][-1] == u"<EOS>"]
            yet_to_gen = [x for x in candidates if x["path"][-1] != u"<EOS>"]
            for candidate in yet_to_gen:
                self.model.predictor.set_state(candidate["cell"], candidate["hidden"])
                prev_word = np.asarray([self.vocab[candidate["path"][-1]]], dtype=np.int32)

                probs = F.softmax(self.model.predictor(prev_word))
                candidates = probs.data[0].argsort()[-1 * width:][::-1]
                cell, hidden = self.model.predictor.get_state()

                for idx in candidates:
                    token = self.ivocab[idx]
                    state = {
                        "path": [x for x in candidate["path"]] + [token],
                        "cell": cell,
                        "hidden": hidden,
                        "prob": candidate["prob"] + np.log(probs.data[0][idx])
                    }
                    temp.append(state)

            candidates = sorted(temp, key=lambda x: x["prob"], reverse=True)[:width]
            if len([x for x in candidates if x["path"][-1] == u"<EOS>"]) == width:
                break
            length += 1
        return " ".join(sorted(temp, key=lambda x: x["prob"], reverse=True)[0]["path"])


def main(args):
    gen = SentenceGenerator(args.vocab, args.model, args.unit)
    print gen.generate(how="beam_search")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Text by RNNLM")
    parser.add_argument('-m', '--model', dest='model', required=True,
                        type=str, help='model data, saved by train_model.py')
    parser.add_argument('-v', '--vocabulary', dest='vocab', required=True,
                        type=str, help='vocabulary data, saved by train_model.py')
    parser.add_argument('-p', '--primetext', dest='prime', default=u'<BOS>',
                        type=str, help='beginning of the sentence')
    parser.add_argument('-l', '--length', dest='length', default=20,
                        type=int, help='length of the generated text')
    parser.add_argument('--gpu', dest='gpu', default=-1,
                        type=int, help='GPU Device ID (Negative Value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int,
                        default=600, help='number of units')
    args = parser.parse_args()
    main(args)
