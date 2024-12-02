#!/usr/bin/env python
# coding: utf-8

from predictionModel import PredictionModel
import tensorflow as tf

import pathlib
import glob
import numpy as np
import pandas as pd
import os

import cv2
from PIL import Image

class makeVideo():
    def __init__(self, model):
        self.model = model
    # outputImagePath : output mp4 name
    def createVideo(self, ImagePathList, outputVideoPath):
        prediction_length = len(ImagePathList)
        prediction_result = self.model.predictionImageList(ImagePathList)
        for index, image_path in enumerate(ImagePathList):
            self.createImage(image_path, prediction_result[index])
        image_path_coord = ImagePathList[0].split('/')
        if len(image_path_coord) > 2:
            predictionImage = sorted( glob.glob("./drawing_result/" + image_path_coord[-2] + "/*.png"))
        else:
            predictionImage = sorted( glob.glob("./drawing_result/*.png"))
        prediction_image_list = []
        for ImagePredict in predictionImage:
            image = cv2.imread(ImagePredict)
            prediction_image_list.append(image)

        outputVideo = cv2.VideoWriter(outputVideoPath, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, self.size)
        for index in range(len(prediction_image_list)):
            outputVideo.write(prediction_image_list[index])
        outputVideo.release()

    def createImage(self, ImagePath, predictPoint):
        path_name = ImagePath.split('/')
        if len(path_name) > 2:
            os.makedirs("./drawing_result/" + path_name[-2], exist_ok=True)
        image = cv2.imread(ImagePath)
        height, width, _ = image.shape[:3]
        cv2.line(image, (int(predictPoint[0][0] * width), int(predictPoint[0][1] * height)),
            (int(predictPoint[0][2] * width), int(predictPoint[0][3] * height)), (255, 255, 255))
        if len(path_name) < 2:
            cv2.imwrite("./drawing_result/" + path_name[-1], image)
        else:
            cv2.imwrite("./drawing_result/" + path_name[-2] + "/" + path_name[-1], image)
        self.size = (width, height)

if __name__ == '__main__':
    # GPU device assignment(tensorflow 2.2.0 ver)
    device_num = 0
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[device_num], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[device_num], True)

    # change parameter
    model = PredictionModel("CCM3",
            "./weight_model/output_weight/sakaki_data_Augument_epoch100/sakaki_data_Augument_epoch100")
    makeV = makeVideo(model)
    image_path_list = sorted(glob.glob("./data/sakaki_data_1/*"))
    # print(image_path_list)
    makeV.createVideo(image_path_list, "./movie/prediction_sakaki_data_1.mp4")
    # image_path_list = sorted(glob.glob("./data/sakaki_data_1/*.jpg"))
    # prediction_result = model.predictionImageList(image_path_list)

