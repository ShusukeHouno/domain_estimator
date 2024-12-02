#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf

def setting_tensorflow(device_num):
    device = device_num
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[device], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[device], True)