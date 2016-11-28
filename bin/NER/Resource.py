# -*- coding: utf-8 -*-
import json
import sys
import os
from datetime import datetime
from pprint import pprint


class Resource(object):
    def __init__(self, config_path):
        time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        with open(config_path, "r") as fi:
            self.config = json.load(fi)
        sys.stderr.write("*** Hyper Prameters ***\n")
        pprint(self.config, stream=sys.stderr)

        self.config["train_data_path"] = self.config['input'] + "train.txt"
        self.config["dev_data_path"] = self.config['input'] + "dev.txt"
        self.config["test_data_path"] = self.config['input'] + "test.txt"
        self.config["vocab_path"] = self.config['input'] + "vocab.txt"
        self.config["time"] = time

        os.mkdir("../result/" + time)
        self.config["log_path"] = "../result/" + time + "/stats.json"

        # snapshot用のディレクトリを準備
        os.mkdir("../result/" + time + "/model_files/")

    def get_data_source(self):
        return self.config["input"]

    def get_device_id(self):
        return self.config["use_gpu"]
