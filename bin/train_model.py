# -*- coding: utf-8 -*-
import argparse
from Trainer import Trainer
from Resource import Resource


def main():
    parser = argparse.ArgumentParser(description="RNN Language Model")
    parser.add_argument('-c', '--config', dest='config', default="../../config/test.json",
                        type=str, help='path to configuration file')
    args = parser.parse_args()
    config = Resource(args.config).config

    trainer = Trainer(config)
    trainer.run()

if __name__ == "__main__":
    main()
