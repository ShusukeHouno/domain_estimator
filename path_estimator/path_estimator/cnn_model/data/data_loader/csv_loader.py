#!/usr/bin/env python
# codeing: utf-8

# TODO: Create by using inheritance.

from email.mime import image
import numpy as np
import pandas as pd
import tensorflow as tf

import csv
from PIL import Image


class CSVLoader:
    def __init__(
        self,
        dataset_path: str,
        flipping_flag: bool,
        batch_size: int,
        Shuffle: int,
        max_branch_num: int,
        Train: bool,
        is_only_point_dataset: bool,
    ):
        """_summary_
        Generate tf Dataset from dataset csv file.

        Args:
            dataset_path (str): dataset path (it's not raw one, pair of img and points representing line)
            flipping_flag (bool): augment dataset by flipping randomly
            batch_size (int): batch size
            Shuffle (int): ref(https://blog.amedama.jp/entry/tf-dataset-api#Datasetshuffle)
            Train (bool): is this dataset for training?
        """
        self.max_branch_size = max_branch_num
        self.is_only_point_dataset = is_only_point_dataset
        # self.img_path_header = "image_path"
        # self.

        self.image_path_list, self.path_list = self.load(dataset_path)
        self.data_size = len(self.image_path_list)
        print(self.path_list)

        # generate dataset
        image_data = tf.data.Dataset.from_tensor_slices(self.image_path_list)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        image_ds = image_data.map(
            self.load_and_preprocess_image, num_parallel_calls=AUTOTUNE
        )
        label_ds = tf.data.Dataset.from_tensor_slices(
            tf.cast(self.path_list, tf.float64)
        )

        print("label_ds {}".format(label_ds))
        image_and_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

        if flipping_flag:
            image_and_label_ds = image_and_label_ds.map(self.rand_flip_tf)

        if Train:
            self.image_and_label_ds = image_and_label_ds.shuffle(Shuffle).batch(
                batch_size
            )
        else:
            self.image_and_label_ds = image_and_label_ds.batch(batch_size)
        print("img_label ds {}".format(self.image_and_label_ds))

    def load(self, dataset_path: str):
        """Load dataset from csv file

        Args:
            dataset_path (str): path(.csv)

        Returns:
            _type_: _description_
        """
        # get coordinates header
        # start_coords_0, end_coords_0, start_coords_1, end_coords_1
        coords_header = []

        if self.max_branch_size == 1:
            if not self.is_only_point_dataset:
                coords_header = ["start_coordinate", "end_coordinate"]
            else:
                coords_header = ["end_coordinate"]
        else:
            for idx in range(self.max_branch_size):
                if not self.is_only_point_dataset:
                    coords_header.append("start_coordinate_" + str(idx))
                coords_header.append("end_coordinate_" + str(idx))

        print(coords_header)

        image_path_list = pd.read_csv(dataset_path, usecols=["image_path"])
        data_path_list = pd.read_csv(dataset_path, usecols=coords_header)

        # create img list
        image_list = [
            image_path_list["image_path"][idx]
            for idx in range(len(image_path_list["image_path"]))
        ]

        # create label list
        label_list = []
        for idx in range(len(image_path_list["image_path"])):
            label_elem = []
            for head in coords_header:
                label_elem.append(eval(data_path_list[head][idx])[0])
                label_elem.append(eval(data_path_list[head][idx])[1])
            label_list.append(label_elem)

        with open("check.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(label_list)

        return image_list, label_list

    def preprocess_image(self, image):
        """Normalize image

        Args:
            image (_type_): _description_

        Returns:
            tf.image: _description_
        """
        image = tf.image.decode_png(image)
        image = tf.image.resize(image, [128, 240])
        image /= 255.0
        return image

    def load_and_preprocess_image(self, path):
        """_summary_
        As usual, this function is called with image_data.map, so automatically input path(?)
        Args:
            path (str): image data path.

        Returns:
            _type_: _description_
        """
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    # def _flip_aug(self, image, label):
    #     image = tf.image.random_contrast(image, 0.7, 0.9999)
    #     image = tf.image.random_brightness(image, 0.003)
    #     rand = np.random.rand()
    #     if rand > 0.5:
    #         image = image[:, ::-1, :]  #
    #         label = label.numpy()
    #         label = tf.constant([1.0 - label[0], 1.0, 1.0 - label[2], 0.8])
    #         return image, label
    #     return image, label

    def _flip_aug(self, image, label):
        """Impl of tf augmentation
        Flip left and right, with probablity of 50%

        Args:
            image (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        image = tf.image.random_contrast(image, 0.7, 0.9999)
        image = tf.image.random_brightness(image, 0.003)
        rand = np.random.rand()
        if rand > 0.5:
            image = image[:, ::-1, :]
            label = label.numpy()
            flipped_label_list = []

            diff = 4
            if self.is_only_point_dataset:
                diff = 2
            for branch_idx in range(self.max_branch_size):
                flipped_label_list.append(1.0 - label[diff * branch_idx])
                flipped_label_list.append(label[diff * branch_idx + 1])
                if not self.is_only_point_dataset:
                    flipped_label_list.append(1.0 - label[diff * branch_idx + 2])
                    flipped_label_list.append(label[diff * branch_idx + 3])
            label = tf.constant(flipped_label_list)
        return image, label

    @tf.function
    def rand_flip_tf(self, image, label):
        image, label = tf.py_function(
            self._flip_aug, [image, label], [tf.float32, tf.float64]
        )
        return image, label

    def get_image_list(self):
        return self.image_path_list

    def get_data_size(self):
        return self.data_size

    def get_dataset(self):
        return self.image_and_label_ds


if __name__ == "__main__":
    dataset_loader = CSVLoader(
        dataset_path="/create_dataset/sakaki_202212/dataset_csv/train.csv",
        flipping_flag=False,
        batch_size=2,
        Shuffle=32,
        Train=False,
    )
    print(dataset_loader.get_dataset())
