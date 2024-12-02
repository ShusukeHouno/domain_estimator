#!/usr/bin/env python
#coding: utf-8

from settings.setting import setting_tensorflow
from model.predict_model import PredictModel
from data.path_dataloader import DataEncode
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy

import tensorflow as tf

import os
import numpy as np
from PIL import Image, ImageDraw
import time
import gc

class Trainer():
    def __init__(self, 
                 cnn_architecture, 
                 train_ds, 
                 test_ds, 
                 epoch, 
                 batch, 
                 is_clopped=True,
                 is_flipped=True,
                 save_weight=''):
        model = PredictModel(cnn_architecture, estimation=False)
        self.cnn_model = model.get_model()

        self.train_ds = DataEncode(train=True, data_path=train_ds, shuffle_buffer=128, batch_size=batch, flipping_flag=is_flipped)
        self.test_ds  = DataEncode(train=False, data_path=test_ds, shuffle_buffer=0, batch_size=batch, flipping_flag=False, clopping_flag=False)
        
        self.optimizer = SGD(lr=0.009, momentum=0.9, decay=4e-5, nesterov=False)
        self.loss_fn = BinaryCrossentropy(from_logits=True)
        # self.training(self.train_dataset.get_dataset(), self.test_dataset.get_dataset(), epoch)
    
    # @tf.function
    # def train_step(self, input, label, itr, epoch):
    #     with tf.GradientTape() as tape:
    #         output = self.model(input)
    #         loss = self.loss_fn(label, output)
    #         acc = tf.reduce_mean(tf.math.abs(tf.cast(tf.math.sigmoid(output), tf.float64)-label))
    #         grad = tape.gradient(loss, self.model.trainable_variables)
    #         self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

    #         self.writing_tensorboard(loss, acc, "train_batch", itr)
    #     return loss, acc

    # @tf.function
    # def calc_metric(self, input, label):
    #     output = self.model(input)
    #     loss = self.loss_fn(label, output)
    #     acc = tf.reduce_mean(tf.math.abs(
    #         tf.cast(tf.math.sigmoid(output), tf.float64) - label))
    #     return loss, acc
    
    def train(self, train_ds, test_ds, epochs):
        for train