#!/usr/bin/env python
#coding: utf-8

import numpy as np
import tensorflow as tf

class NPZLoader():
    def __init__(self, dataset_path, flipping_flag, batch_size, shuffle, Train):
        self.npz_path_list= self.npz_list_loader(dataset_path)
        self.image_path_list, self.path_list = self.convert_label(self.npz_path_list)
        self.data_size = len(self.image_path_list)

        self.image_and_label_ds = self.create_dataset(self.image_path_list, self.path_list)
        if Train:
            if flipping_flag:
                self.image_and_label_ds = self.image_and_label_ds.map(self.rand_flip_tf).shuffle(shuffle).batch(batch_size)
            else:
                self.image_and_label_ds = self.image_and_label_ds.shuffle(shuffle).batch(batch_size)
        else:
            self.image_and_label_ds = self.image_and_label_ds.batch(batch_size)

    def create_dataset(self, image_data, path_data):
        tf_image_path = tf.data.Dataset.from_tensor_slices(image_data)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        tf_image = tf_image_path.map(self.load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(path_data, tf.float64))
        image_and_label = tf.data.Dataset.zip((tf_image, label_ds))
        return image_and_label

    def convert_label(self, binary_image_list):
        path_data, image_path = [], []
        for data in binary_image_list:
            tuple_data = []
            # initialize binary data to 0 or 1
            binary_data = self.encode(data)/255
            # get image data size
            height, width = binary_data.shape
            # Where is the location of the final data?
            height_data = np.where(np.any(binary_data == 1, axis=1) == 1)
            way_point = height_data[0][height_data[0] == int(height * 0.8)]
            if len(way_point) == 0:
                continue
            # Trajectory data list creation
            tuple_data.append(int(np.average(np.where(binary_data[height_data[0][-1]] == 1))) / width)
            tuple_data.append(1.0)
            tuple_data.append(int(np.average(np.where(binary_data[way_point] == 1)[1])) / width)
            tuple_data.append(0.8)

            # List the paths of the data to be used
            image_path.append(data.replace(".npz", ".png"))
            path_data.append(tuple_data)

        return image_path, path_data


    def encode(self, binary_data):
        # load npz-file to image array
        npz_mat = np.load(binary_data)
        return npz_mat['arr_0']

    def npz_list_loader(self, dataset_path):
        npz_list = []
        with open(dataset_path) as file:
            for line in file:
                npz_list.append(line.replace("\n", ""))
        return npz_list

    # augumentation flip function
    def _flip_aug(self, image, label):
        image = tf.image.random_contrast(image, 0.7, 0.9999)
        image = tf.image.random_brightness(image, 0.003)
        rand = np.random.rand()
        if rand > 0.5:
            image = image[:, ::-1, :]
            label_ = label.numpy()
            label_ = tf.constant([1.0 - label_[0], label_[1],
                                1.0 - label_[2], label_[3]])
            return image, label_
        return image, label

    @tf.function
    def rand_flip_tf(self, image, label):
        image, label = tf.py_function(self._flip_aug, [image, label], [tf.float32, tf.float64])
        return image, label

    # resize image data
    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [128, 240])
        image /= 255.0
        return image

    # decord image pass to image
    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def get_image_list(self):
        with open('dataset_list.txt', "w") as file_data:
            for data in self.image_path_list:
                file_data.write("%s\n" % data)
        return self.image_path_list

    def get_path_list(self):
        return self.path_list

    def get_data_size(self):
        return self.data_size

    def get_dataset(self):
        return self.image_and_label_ds

if __name__=="__main__":
    npz_loader = NPZLoader(dataset_path="/create_dataset/sakaki_202212/dataset_npz/test.txt", flipping_flag=True, batch_size=2, shuffle=32, Train=False)
    print(npz_loader.get_dataset())
