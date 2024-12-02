#!/usr/bin/env python
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dense,
    Activation,
    Dropout,
    Flatten,
)
from tensorflow.keras.optimizers import Adam


class PathEstimationCCP_JCT:
    def __init__(self, weight_data="", estimation=False):
        self.model = self.set_model_layer()
        if weight_data:
            self.update_weight(weight_data)
        if estimation:
            self.model = tf.keras.Sequential(
                [self.model, tf.keras.layers.Activation("sigmoid")]
            )
        self.summary_model()

    def set_model_layer(self):
        """_summary_
        Creation of estimator
        """
        model = Sequential()
        model.add(
            Conv2D(
                32,
                3,
                strides=(1, 1),
                padding="same",
                activation="relu",
                input_shape=(128, 240, 3),
            )
        )
        # kernel 3x3
        model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(2))

        return model

    def update_weight(self, weight_path):
        """_summary_
        Args:
            weight_path (string): model weight(pass)
        """
        print("load weight from {}".format(weight_path))
        self.model.load_weights(weight_path)

    def summary_model(self):
        """_summary_
        Output description
        """
        return
        with open("/home/nanoshimarobot/sandbox_ws/tf/training_summary.txt", "w") as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

    def get_model(self):
        return self.model

    def save_model(self, weight_name):
        self.model.save_weights(weight_name)


class PathEstimationCCCP_MLT2:
    def __init__(self, weight_data="", estimation=False):
        self.model = self.set_model_layer()
        if weight_data:
            self.update_weight(weight_data)
        if estimation:
            self.model = tf.keras.Sequential(
                [self.model, tf.keras.layers.Activation("sigmoid")]
            )
        self.summary_model()

    def set_model_layer(self):
        """_summary_
        Creation of estimator
        """
        model = Sequential()
        model.add(
            Conv2D(
                32,
                3,
                strides=(1, 1),
                padding="same",
                activation="relu",
                input_shape=(128, 240, 3),
            )
        )
        # kernel 3x3
        model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(8))
        # self.summary_model()

        return model

    def update_weight(self, weight_pass):
        """_summary_
        Args:
            weight_pass (string): model weight(pass)
        """
        print("load weight from {}".format(weight_pass))
        self.model.load_weights(weight_pass)

    def summary_model(self):
        """_summary_
        Output description
        """
        return
        with open("/home/nanoshimarobot/sandbox_ws/tf/training_summary.txt", "w") as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

    def get_model(self):
        return self.model

    def save_model(self, weight_name):
        self.model.save_weights(weight_name)


class PathEstimationCCP_MLT2:
    def __init__(self, weight_data="", estimation=False):
        self.model = self.set_model_layer()
        if weight_data:
            self.update_weight(weight_data)
        if estimation:
            self.model = tf.keras.Sequential(
                [self.model, tf.keras.layers.Activation("sigmoid")]
            )
        self.summary_model()

    def set_model_layer(self):
        """_summary_
        Creation of estimator
        """
        model = Sequential()
        model.add(
            Conv2D(
                32,
                3,
                strides=(1, 1),
                padding="same",
                activation="relu",
                input_shape=(128, 240, 3),
            )
        )
        # kernel 3x3
        model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(8))
        # self.summary_model()

        return model

    def update_weight(self, weight_pass):
        """_summary_
        Args:
            weight_pass (string): model weight(pass)
        """
        print("load weight from {}".format(weight_pass))
        self.model.load_weights(weight_pass)

    def summary_model(self):
        """_summary_
        Output description
        """
        return
        with open("/home/nanoshimarobot/sandbox_ws/tf/training_summary.txt", "w") as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

    def get_model(self):
        return self.model

    def save_model(self, weight_name):
        self.model.save_weights(weight_name)


class PathEstimationCCP_MLT2_TANH:
    def __init__(self, weight_data="", estimation=False):
        self.model = self.set_model_layer()
        if weight_data:
            self.update_weight(weight_data)
        if estimation:
            self.model = tf.keras.Sequential(
                [self.model, tf.keras.layers.Activation("sigmoid")]
            )
        self.summary_model()

    def set_model_layer(self):
        """_summary_
        Creation of estimator
        """
        model = Sequential()
        model.add(
            Conv2D(
                32,
                3,
                strides=(1, 1),
                padding="same",
                activation="tanh",
                input_shape=(128, 240, 3),
            )
        )
        # kernel 3x3
        model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="tanh"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(128, 3, strides=(1, 1), padding="same", activation="tanh"))
        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="tanh"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="tanh"))
        model.add(Conv2D(512, 3, strides=(1, 1), padding="same", activation="tanh"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation="tanh"))
        model.add(Dense(8))
        # self.summary_model()

        return model

    def update_weight(self, weight_pass):
        """_summary_
        Args:
            weight_pass (string): model weight(pass)
        """
        print("load weight from {}".format(weight_pass))
        self.model.load_weights(weight_pass)

    def summary_model(self):
        """_summary_
        Output description
        """
        return
        with open("/home/nanoshimarobot/sandbox_ws/tf/training_summary.txt", "w") as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

    def get_model(self):
        return self.model

    def save_model(self, weight_name):
        self.model.save_weights(weight_name)


class PathEstimationCCP:
    def __init__(self, weight_data="", estimation=False):
        self.model = self.set_model_layer()
        if weight_data:
            self.update_weight(weight_data)
        if estimation:
            self.model = tf.keras.Sequential(
                [self.model, tf.keras.layers.Activation("sigmoid")]
            )
        self.summary_model()

    def set_model_layer(self):
        """_summary_
        Creation of estimator
        """
        model = Sequential()
        model.add(
            Conv2D(
                32,
                3,
                strides=(1, 1),
                padding="same",
                activation="relu",
                input_shape=(128, 240, 3),
            )
        )
        model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(4))
        # self.summary_model()

        return model

    def update_weight(self, weight_pass):
        """_summary_
        Args:
            weight_pass (string): model weight(pass)
        """
        print("load weight from {}".format(weight_pass))
        self.model.load_weights(weight_pass)

    def summary_model(self):
        """_summary_
        Output description
        """
        return
        with open("/home/nanoshimarobot/sandbox_ws/tf/training_summary.txt", "w") as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

    def get_model(self):
        return self.model

    def save_model(self, weight_name):
        self.model.save_weights(weight_name)


class PathEstimationCP:
    def __init__(self, weight_data="", estimation=False):
        self.model = Sequential()
        self.set_model_layer()
        if weight_data:
            self.update_weight(weight_data)
        if estimation:
            self.model = tf.keras.Sequential(
                [self.model, tf.keras.layers.Activation("sigmoid")]
            )
        self.summary_model()

    def set_model_layer(self):
        """_summary_
        Creation of estimator
        """
        self.model.add(
            Conv2D(
                32,
                3,
                strides=(1, 1),
                padding="same",
                activation="relu",
                input_shape=(128, 240, 3),
            )
        )
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(
            Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu")
        )
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(
            Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu")
        )
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(4))

    def update_weight(self, weight_pass):
        """_summary_
        Args:
            weight_pass (string): model weight(pass)
        """
        self.model.load_weights(weight_pass)

    def summary_model(self):
        """_summary_
        Output description
        """
        return
        with open("/tf/log/training_summary.txt", "w") as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

    def get_model(self):
        return self.model


if __name__ == "__main__":
    model = PathEstimationCCP()
