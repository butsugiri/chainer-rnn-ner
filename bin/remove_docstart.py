# -*- coding: utf-8 -*-
"""

"""
import sys

def main(fi):
    docstart_flag = False
    for line in fi:
        if "-DOCSTART-" in line:
            line.strip()
            docstart_flag = True
            continue
        if docstart_flag:
            docstart_flag = False
            continue
        print(line.strip())

if __name__ == "__main__":
    main(sys.stdin)
