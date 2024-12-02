#!/usr/bin/env python3
#coding:utf-8

from model.predict_model import PredictModel
from settings.setting import setting_tensorflow

import math
import numpy as np
import cv2
import copy
import csv
import random
from PIL import Image, ImageDraw

if __name__=="__main__":
    setting_tensorflow(0)
    cnn_model = PredictModel('CCP', '/tf/weight/epoch100',True)
    model = cnn_model.get_model()

    data_list = []
    with open('/generated_dataset/sakaki_for_validation/generated_train_dataset.csv', "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for r_idx, row in enumerate(reader):
                row_data = []
                if r_idx != 0:
                    for idx, data in enumerate(row):
                        if data == "":
                            continue
                        if idx != 0:
                            data = eval(data)
                        row_data.append(data)
                    data_list.append(row_data)
    
    for i in range(10):
        output_path = "/generated_dataset/sakaki_for_validation/test" + str(i) + ".png"
        target_idx = random.randint(0, len(data_list)-1)
        target_record = data_list.pop(target_idx)
        target_img_path = target_record[0]
        img = cv2.imread(target_img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = cv2.resize(img, (240, 128))
        
        img_np = img[:, :, ::-1].copy()
        img_np = cv2.resize(img_np, (240, 128))
        img_np_norm = np.array(img_np) / 255.0
        img_np_norm = img_np_norm[np.newaxis, ...]
        
        print(img_np_norm)
        
        height_, width_, _ = img.shape
        r_height_, r_width_, _ = img_norm.shape

        path = model.predict(img_np_norm)
        # print(path)
        path = np.array(((path[0][0] * width_), (path[0][1] * height_), \
                       (path[0][2] * width_), (path[0][3] * height_)), dtype=np.uint16)
        print(path)
        # output_image = cv2.line(img_norm,
        # #                     # pt1=(int(path[0][0] * width_), int(path[0][1] * height_)),
        # #                     # pt2=(int(path[0][2] * width_), int(path[0][3] * height_)),
        #                     pt1=(int(path[0] / width_ * r_width_), int(path[1] / height_ * r_height_)),
        #                     pt2=(int(path[2] / width_ * r_width_), int(path[3] / height_ * r_height_)),
        #                     color=(0, 0, 255),
        #                     thickness=3)
        output_image = cv2.line(img,
                                pt1=(int(path[0]), int(path[1])),
                                pt2=(int(path[2]), int(path[3])),
                                color=(0, 0, 255),
                                thickness=3)
        cv2.imwrite(output_path, output_image)

    # for i in range(100):
    #     input_path = "/raw_/sakaki_for_validation/" + str(i).zfill(4) + ".png"
    #     output_path = "/raw_dataset/output_img/out1_" + str(i).zfill(4) + ".png"
    #     image = cv2.imread(input_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     height_, width_, _ = image.shape
    #     image = cv2.resize(image, (240, 128))

    #     np_image = image[:, :, ::-1].copy()
    #     np_image = cv2.resize(np_image, (240, 128))
    #     np_image = np.array(np_image) / 255.0

    #     np_image = np_image[np.newaxis, ...]

    #     r_height_, r_width_, _ = image.shape

    #     path = model.predict(np_image)
    #     # print(path)
    #     path = np.array(((path[0][0] * width_), (path[0][1] * height_), \
    #                    (path[0][2] * width_), (path[0][3] * height_)), dtype=np.uint16)
    #     print(path)
    #     output_image = cv2.line(image,
    #     #                     # pt1=(int(path[0][0] * width_), int(path[0][1] * height_)),
    #     #                     # pt2=(int(path[0][2] * width_), int(path[0][3] * height_)),
    #                         pt1=(int(path[0] / width_ * r_width_), int(path[1] / height_ * r_height_)),
    #                         pt2=(int(path[2] / width_ * r_width_), int(path[3] / height_ * r_height_)),
    #                         color=(0, 0, 255),
    #                         thickness=3)
    
    #     # cv2.imwrite(output_path, output_image)