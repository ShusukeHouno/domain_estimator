#!/usr/bin/env python
# coding: utf-8

# Initial setup for using tensorflow
from distutils.command.install_egg_info import to_filename
from settings.setting import setting_tensorflow
from model.predict_model import PredictModel
from data.path_dataloader import DataEncode

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy

# from keras.losses import BinaryCrossentropy

import tqdm
import os
import numpy as np
from PIL import Image, ImageDraw
import time
import gc


class Train:
    def __init__(
        self,
        model_architecture: str,
        model_name: str,
        train_dataset: str,
        validation_dataset: str,
        epoch,
        batch: str,
        max_branch_num=0,
        is_only_point_dataset=False,
        flipping_flag=True,
    ):
        """_summary_
        Get CNN model and train based on the train-dataset and test-dataset
        Args:
            model_architecture (str): CP or CCP, different architecture
            model_name (str) : CNN model name.
            train_dataset (str): train dataset path
            validation_dataset (str): test dataset path
            epoch (_type_): _description_
            batch (_type_): _description_
            clopping_flag (bool, optional): _description_. Defaults to True.
            flipping_flag (bool, optional): _description_. Defaults to True.
            save_weight (str, optional): _description_. Defaults to "".
        """
        self.max_branch_num = max_branch_num
        self.is_only_point_dataset = is_only_point_dataset
        model = PredictModel(model_architecture, estimation=False)
        self.model_architecture = model_architecture
        self.model_name = model_name

        self.model = model.get_model()
        self.train_dataset = DataEncode(
            train=True,
            data_path=train_dataset,
            shuffle_buffer=128,
            batch_size=batch,
            max_branch_num=max_branch_num,
            is_only_point_dataset=is_only_point_dataset,
            flipping_flag=flipping_flag,
        )
        self.validation_dataset = DataEncode(
            train=False,
            data_path=validation_dataset,
            shuffle_buffer=0,
            batch_size=batch,
            max_branch_num=max_branch_num,
            is_only_point_dataset=is_only_point_dataset,
            flipping_flag=False,
            clopping_flag=False,
        )

        # return

        self.tensorboard_folder(
            model_architecture, model_name, epoch, batch, flipping_flag
        )

        # optimizer and loss function
        self.optimizer = SGD(lr=0.009, momentum=0.9, nesterov=False)
        # self.optimizer = SGD(lr=0.009, momentum=0.9, decay=4e-5, nesterov=False)
        self.loss_fn = BinaryCrossentropy(from_logits=True)

        self.training(
            self.train_dataset.get_dataset(),
            self.validation_dataset.get_dataset(),
            epoch,
        )

    # make tensorboard folder (checked)
    def tensorboard_folder(
        self, model_architecture, model_name, epoch, batch, flipping_flag
    ):
        """Make tensorboard directory

        Args:
            model_architecture (_type_): _description_
            model_name (_type_): _description_
            epoch (_type_): _description_
            batch (_type_): _description_
            flipping_flag (_type_): _description_
        """
        model_data = (
            model_name
            + "_"
            + model_architecture
            + "_epc"
            + str(epoch)
            + "_batch"
            + str(batch)
        )
        if flipping_flag:
            model_data = model_data + "_with_flip"
        log_dir = os.path.join("/tf/tensorboard/log/", model_data)
        self.summary_train_writer = tf.summary.create_file_writer(log_dir + "fit/train")
        self.summary_validation_writer = tf.summary.create_file_writer(
            log_dir + "fit/validation"
        )
        self.summary_image_writer = tf.summary.create_file_writer(log_dir + "fit/image")
        self.summary_image_writer_pre = tf.summary.create_file_writer(
            log_dir + "fit/estimate_image"
        )

    @tf.function
    def train_step(self, input, label, itr):
        """Training (one-step)

        Args:
            input (tensor): image tensor
            label (tensor): path-label tensor
            itr (int): _description_

        Returns:
            _type_: _description_
        """
        with tf.GradientTape() as tape:
            output = self.model(input)
            loss = self.loss_fn(label, output)
            # 各点のズレの平均の、バッチ全体の平均
            acc = tf.reduce_mean(
                tf.math.abs(tf.cast(tf.math.sigmoid(output), tf.float64) - label)
            )
            grad = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

            self.writing_tensorboard(loss, acc, "train_batch", itr)
        return loss, acc

    @tf.function
    def calc_metric(self, input, label):
        """_summary_
        For evaluating model. Loss and accuracy were calculated in here
        Args:
            input (tensor): shape
            label (_type_): _description_

        Returns:
            _type_: loss, acc
        """
        output = self.model(input)
        loss = self.loss_fn(label, output)
        acc = tf.reduce_mean(
            tf.math.abs(tf.cast(tf.math.sigmoid(output), tf.float64) - label)
        )
        return loss, acc

    def training(self, train_ds, validation_ds, epochs):
        event_time = epochs // 20
        if event_time <= 0:
            event_time = 1
        step = 0
        for epoch in range(epochs):
            start = time.time()
            epoch_tf = tf.convert_to_tensor(epoch, dtype=tf.int64)
            loss_list = []
            acc_list = []

            # train using all of batch
            for n, (input, label) in train_ds.enumerate():
                # print(input)
                if n == 0:
                    image_data = np.empty((1, 128 * 3, 240 * 4, 3), dtype=np.uint8)
                step_tf = tf.convert_to_tensor(step, dtype=tf.int64)
                if 0 == epoch % event_time:
                    input_images = self.input_image_drawing(
                        input, label, is_drawing=False
                    )
                    if not type(input_images) is bool:
                        image_data = np.vstack((image_data, input_images))
                loss_step, acc_step = self.train_step(input, label, step_tf)
                step += 1
                loss_list.append(loss_step)
                acc_list.append(acc_step)

            if 0 == epoch % event_time:
                image_data = np.delete(image_data, 0, axis=0)
                with self.summary_image_writer.as_default():
                    tf.summary.image(
                        "input_image", image_data, step=epoch, max_outputs=5
                    )

            ave_train_loss = np.average(loss_list)
            ave_train_acc = np.average(acc_list)
            tensor_train_loss = tf.convert_to_tensor(ave_train_loss, dtype=tf.float32)
            tensor_train_acc = tf.convert_to_tensor(ave_train_acc, dtype=tf.float32)
            self.writing_tensorboard(
                tensor_train_loss, tensor_train_acc, "train_epoch", epoch_tf
            )

            loss_list = []
            acc_list = []

            # validation step
            for n, (input, label) in validation_ds.enumerate():
                if n == 0:
                    estimate_data = np.empty((1, 128 * 3, 240 * 4, 3), dtype=np.uint8)
                loss_step, acc_step = self.calc_metric(input, label)
                if 0 == epoch % event_time:
                    estimate_image = self.input_image_drawing(
                        input, label, is_drawing=True
                    )
                    if not type(estimate_image) is bool:
                        estimate_data = np.vstack((estimate_data, estimate_image))
                loss_list.append(loss_step.numpy())
                acc_list.append(acc_step.numpy())

            if 0 == epoch % event_time:
                estimate_data = np.delete(estimate_data, 0, axis=0)
                with self.summary_image_writer_pre.as_default():
                    tf.summary.image(
                        "estimation_image", estimate_data, step=epoch, max_outputs=5
                    )

            ave_validation_loss = np.average(loss_list)
            ave_validation_acc = np.average(acc_list)
            tensor_validation_loss = tf.convert_to_tensor(
                ave_validation_loss, dtype=tf.float32
            )
            tensor_validation_acc = tf.convert_to_tensor(
                ave_validation_acc, dtype=tf.float32
            )
            self.writing_tensorboard(
                tensor_validation_loss,
                tensor_validation_acc,
                "validation_epoch",
                epoch_tf,
            )

            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, time.time() - start
                )
            )
            gc.collect()
        self.model.save_weights(
            "/tf/weight/"
            + self.model_name
            + "_"
            + self.model_architecture
            + "_epc"
            + str(epochs)
        )

    def input_image_drawing(self, image, label, is_drawing=False):
        image_ = image.numpy()
        background_image = Image.new("RGB", (240 * 4, 128 * 3))
        if len(image_) < 12:
            return False
        for index in range(12):
            prot_data = image_[index] * 255
            pil_data = Image.fromarray(np.uint8(prot_data))
            row, col = divmod(index, 4)
            if is_drawing:
                predict_data = image_[index]
                estimate_data = self.model(predict_data[np.newaxis, ...]).numpy()
                if not self.is_only_point_dataset:
                    self.estimation_drawing(
                        background_image,
                        row,
                        col,
                        pil_data,
                        label[index],
                        estimate_data,
                    )
                else:
                    self.estimation_point_draw(
                        background_image,
                        row,
                        col,
                        pil_data,
                        label[index],
                        estimate_data,
                    )
            else:
                background_image.paste(pil_data, (240 * col, 128 * row))
        write_image_data = np.asarray(background_image)
        write_image_data = write_image_data[np.newaxis, ...]
        return write_image_data

    def estimation_point_draw(self, back_img, row, col, image, label, estimate):
        image_canvas = ImageDraw.Draw(image)
        width, height = image.width, image.height
        for idx in range(self.max_branch_num):
            truth_coords = np.array(
                (
                    label[2 * idx + 0] * width - 5,
                    label[2 * idx + 1] * height - 5,
                    label[2 * idx + 0] * width + 5,
                    label[2 * idx + 1] * height + 5,
                ),
            )

            estimate_coords = np.array(
                (
                    self.sigmoid(estimate[0][2 * idx + 0]) * width - 5,
                    self.sigmoid(estimate[0][2 * idx + 1]) * height - 5,
                    self.sigmoid(estimate[0][2 * idx + 0]) * width + 5,
                    self.sigmoid(estimate[0][2 * idx + 1]) * height + 5,
                )
            )

            image_canvas.ellipse(truth_coords.tolist(), fill=(255, 0, 0), width=3)
            image_canvas.ellipse(estimate_coords.tolist(), fill=(0, 255, 255), width=3)
        back_img.paste(image, (240 * col, 128 * row))

    def estimation_drawing(self, back_image, row, col, image, label, estimate):
        image_canvas = ImageDraw.Draw(image)
        width, height = image.width, image.height

        for idx in range(self.max_branch_num):
            truth_coords = np.array(
                (
                    label[4 * idx + 0] * width,
                    label[4 * idx + 1] * height,
                    label[4 * idx + 2] * width,
                    label[4 * idx + 3] * height,
                ),
                dtype=np.uint8,
            )

            estimate_coords = np.array(
                (
                    self.sigmoid(estimate[0][4 * idx + 0]) * width,
                    self.sigmoid(estimate[0][4 * idx + 1]) * height,
                    self.sigmoid(estimate[0][4 * idx + 2]) * width,
                    self.sigmoid(estimate[0][4 * idx + 3]) * height,
                ),
                dtype=np.uint8,
            )

            image_canvas.line(truth_coords.tolist(), fill=(255, 0, 0), width=3)
            image_canvas.line(estimate_coords.tolist(), fill=(0, 255, 255), width=3)
        back_image.paste(image, (240 * col, 128 * row))

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def writing_tensorboard(self, loss, error, name, step):
        if "train" in name:
            with self.summary_train_writer.as_default():
                tf.summary.scalar(name + "_loss", loss, step=step)
                tf.summary.scalar(name + "_error", error, step=step)
        elif "validation" in name:
            with self.summary_validation_writer.as_default():
                tf.summary.scalar(name + "_loss", loss, step=step)
                tf.summary.scalar(name + "_error", error, step=step)


if __name__ == "__main__":
    setting_tensorflow(device_num=0)
    # train_module = Train(
    #     model_architecture="CCP",
    #     model_name="TESTMULT",
    #     train_dataset="/generated_dataset/sakaki_ds_l2_202310160055/generated_train_dataset.csv",
    #     validation_dataset="/generated_dataset/sakaki_ds_l2_202310160055/generated_validation_dataset.csv",
    #     epoch=100,
    #     batch=32,
    #     max_branch_num=1
    # )
    train_module = Train(
        model_architecture="CCP",
        model_name="TESTMULTPOINTS",
        train_dataset="/generated_dataset/TESTsakaki_r9_l15_multi_point202311100029/generated_train_dataset.csv",
        validation_dataset="/generated_dataset/TESTsakaki_r9_l15_multi_point202311100029/generated_test_dataset.csv",
        epoch=1,
        batch=32,
        max_branch_num=2,
        is_only_point_dataset=True,
    )
