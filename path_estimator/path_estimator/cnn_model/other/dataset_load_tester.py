#!/usr/bin/env python
#coding: utf-8

from distutils.command.install_egg_info import to_filename
from settings.setting import setting_tensorflow
from model.predict_model import PredictModel
from data.path_dataloader import DataEncode

import tensorflow as tf

if __name__=="__main__":
    train_ds = DataEncode(train=True, data_path = "/generated_dataset/sakaki_trial1_dataset/generated_dataset.csv", shuffle_buffer=128,
                                        batch_size=32, flipping_flag=False)
    
    print(train_ds.get_dataset)
    # for (input, label) in train_ds.get_dataset().enumerate():
    #     print(input,label)
    #     break