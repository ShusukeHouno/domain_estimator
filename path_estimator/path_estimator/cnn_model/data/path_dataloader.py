#!/usr/bin/env python
# coding: utf-8

from data.data_loader.csv_loader import CSVLoader
from data.data_loader.npz_loader import NPZLoader

# from data_loader.csv_loader import CSVLoader
# from data_loader.npz_loader import NPZLoader

import pandas as pd
import tensorflow as tf
from PIL import Image
import time


class DataEncode:
    def __init__(
        self,
        train=True,
        data_path="./dataset_path_file/path",
        shuffle_buffer=128,
        batch_size=32,
        max_branch_num=0,
        is_only_point_dataset=False,
        clopping_flag=False,
        flipping_flag=True,
    ):
        if ".csv" in data_path:
            print("csv mode")
            self.dataset, self.dataset_size = self.load_csv(
                data_path,
                flipping_flag,
                batch_size,
                shuffle_buffer,
                max_branch_num,
                train,
                is_only_point_dataset
            )
        elif ".txt" in data_path:
            print("npz load mode")
            self.dataset, self.dataset_size = self.load_npz(
                data_path, flipping_flag, batch_size, shuffle_buffer, train
            )
        else:
            print("[error!] select file! ex:.csv, .npz")
        print(self.dataset_size)

    def load_csv(
        self,
        data_path,
        flipping_flag,
        batch_size,
        shuffle_buffer,
        max_branch_num,
        train,
        is_only_point_dataset
    ):
        load_dataset = CSVLoader(
            data_path, flipping_flag, batch_size, shuffle_buffer, max_branch_num, train, is_only_point_dataset
        )
        return load_dataset.get_dataset(), load_dataset.get_data_size()

    def load_npz(self, data_path, flipping_flag, batch_size, shuffle_buffer, train):
        load_dataset = NPZLoader(
            data_path, flipping_flag, batch_size, shuffle_buffer, train
        )
        load_dataset.get_image_list()
        return load_dataset.get_dataset(), load_dataset.get_data_size()

    def get_dataset(self):
        return self.dataset
