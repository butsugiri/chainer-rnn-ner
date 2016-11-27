# -*- coding: utf-8 -*-
import argparse
import sys
import json
from collections import defaultdict

def main(fi, threshold):
    token_freqs = defaultdict(int)
    pos_tags = set()
    for line in fi:
        tokens = json.loads(line)
        for token in tokens:
            token_freqs[token["surface"]] += 1
            pos_tags.add(token["pos"])

    token_ids = defaultdict(lambda: len(token_ids))
    for tag in pos_tags:
        token_ids['{}<UNK>'.format(tag)]

    for token, freq in token_freqs.items():
        if freq <= threshold:
            continue
        elif token.strip() == "":
            continue
        else:
            token_ids[token]

    for vocab, _id in token_ids.items():
        print("{}\t{}".format(vocab, _id))

    sys.stderr.write("Threshold Value: {}\nOriginal Vocab Size:{}\tVocab Size (After Cut-off):{}\n".format(
        threshold,
        len(token_freqs),
        len(token_ids),
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Vocab creator")
    parser.add_argument('-t', '--threshold', dest='threshold', default=2, type=int,help='しきい値')
    args = parser.parse_args()
    main(sys.stdin, args.threshold)
