#!/bin/sh
python preprocess.py < ../data/train > ../work/train.clean
python preprocess.py < ../data/dev > ../work/dev.clean
python preprocess.py < ../data/test > ../work/test.clean
