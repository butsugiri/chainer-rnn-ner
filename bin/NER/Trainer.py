# -*- coding: utf-8 -*-
import chainer
from chainer import training
from .DataProcessor import DataProcessor


class TrainerUI(training.Trainer):

    def __init__(self, resource):
        gpu = resource.get_device_id()
        data_path = resource.get_data_source()
        data_processor = DataProcessor(data_path, gpu)
        data_processor.prepare()
        self.train = data_processor.train_data
        self.dev = data_processor.dev_data

        super(Trainer, self).__init__(
            updater=updater,
            stop_trigger=(total_epoch, 'epoch')
        )
        pass
    pass
