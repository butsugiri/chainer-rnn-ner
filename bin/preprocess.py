# -*- coding: utf-8 -*-
"""
rawデータを標準入力から受け取って一行一文にする
雰囲気としては__token__::__pos__
"""
import sys
import json
from itertools import groupby

def main(fi):
    for is_empty, section in groupby(fi, key=lambda x: x.strip() == ""):
        if not is_empty:
            sent = []
            section = list(section)
            if section[0].split("\t")[1].startswith("-DOCSTART-"):
                continue
            if section[0].split() == "":
                continue
            for line in section:
                target, surface, pos, chunk = line.strip().split("\t")
                token = {
                    "surface": surface.lower(),
                    "pos": pos,
                    "target": target
                }
                sent.append(token)
            print(json.dumps(sent))

if __name__ == '__main__':
    main(sys.stdin)
