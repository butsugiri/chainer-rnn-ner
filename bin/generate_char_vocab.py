# -*- coding: utf-8 -*-
import argparse
import sys
import json
from collections import defaultdict

def main(fi):
    char_ids = defaultdict(lambda: len(char_ids))
    char_ids["<UNK>"]
    for line in fi:
        tokens = json.loads(line)
        for token in tokens:
            for char in token["raw"]:
                char_ids[char]

    for vocab, _id in char_ids.items():
        print("{}\t{}".format(vocab, _id))

if __name__ == "__main__":
    main(sys.stdin)
