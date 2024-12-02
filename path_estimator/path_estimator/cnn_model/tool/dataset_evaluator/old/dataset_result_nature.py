#!/usr/bin/python python3
#encode:utf-8
import sys

sys.path.append('../../')

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from settings import *
import matplotlib.pyplot as plt
import copy
import csv

TARGET_DATASET_PATH = "/Nature_env_dataset/dataset_csv/train.csv"
OBJECT_DATASET_PATH = "/Nature_env_dataset/dataset_npz/train.txt"
#TARGET_DATASET_PATH = "/create_dataset/sakaki_202212/dataset_npz/fix_train_aug_2.txt"
#OBJECT_DATASET_PATH = "/create_dataset/sakaki_202212/dataset_csv/fix_train_aug_3.csv"

# TARGET_DATASET_PATH = "/create_dataset/sakaki_202212/dataset_csv/auto_ann_train.csv"
# OBJECT_DATASET_PATH = "/create_dataset/sakaki_202212/dataset_csv/manual_ann_train.csv"

PDF_DATA = "diff_target_auto_object_manual_nature"
# PDF_DATA = "diff_target_npz_object_csv"
# PDF_DATA = "diff_target_auto_object_manual_greenhouse"

class DataCompare():
    def __init__(self, target_dataset, object_dataset, clop_check=False):
        self.target_dataset = self.dataset_loader(target_dataset, clop_check)
        self.object_dataset = self.dataset_loader(object_dataset, clop_check)
        self.target_dataset = self.data_collect(self.target_dataset, self.object_dataset)
        print(self.target_dataset)
        self.object_dataset = self.data_collect(self.object_dataset, self.target_dataset)
        self.diff_label = self.diff_dataset()

    def dataset_loader(self, dataset_name, clop_check):
        if ".txt" in dataset_name:
            return self.npz_loader(dataset_name, clop_check)
        else:
            return self.csv_loader(dataset_name, clop_check)

    def data_collect(self, target, object):
        new_data = []
        new_coord_data = []
        for index, path_data in enumerate(target[0]):
            if path_data in object[0]:
                new_data.append(path_data)
                new_coord_data.append(target[1][index])
        return (new_data, new_coord_data)

    def diff_dataset(self):
        self.image_path = [[self.target_dataset[0][index],self.object_dataset[0][index]] for index in range(len(self.target_dataset[0]))]
        with open("check.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(self.image_path)
        self.start_x = [self.target_dataset[1][index][0] - self.object_dataset[1][index][0]
                        for index in range(len(self.target_dataset[0]))]
        self.end_x = [self.target_dataset[1][index][2] - self.object_dataset[1][index][2]
                        for index in range(len(self.target_dataset[0]))]
        print(self.end_x)
        self.start_x = np.array(self.start_x)
        self.end_x = np.array(self.end_x)

        bins = [-0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        start_coordinate_data, start_data_bins = np.histogram(self.start_x, range=(-0.65, 0.65), density=True, bins=bins)
        end_coordinate_data, end_data_bins = np.histogram(self.end_x, range=(-0.65, 0.65), density=True, bins=bins)
        fig = plt.figure(figsize=(10, 5))
        start_graph = fig.add_subplot(1, 2, 1)
        end_graph = fig.add_subplot(1, 2, 2)
        self.setting_graph(start_graph, start_coordinate_data, start_data_bins, "start")
        self.setting_graph(end_graph, end_coordinate_data, end_data_bins, "end")
        plt.savefig("pdf/" + PDF_DATA + ".pdf", format="pdf", dpi=300)

    # dataset loader
    def csv_loader(self, file_name, clop_check):
        image_data = pd.read_csv(file_name, usecols=['image_path'])
        coord_data = pd.read_csv(file_name, 
                                        usecols=['start_coordinate', 'end_coordinate'])
        # coord_data = pd.read_csv(file_name, 
        #                                 usecols=['x_start', 'y_start', 'x_end', 'y_end'])
        range_data = len(image_data['image_path'])
        raw_data = copy.deepcopy(image_data)
        if clop_check:
            image_data = [image_data['image_path'][index]
                                for index in range(len(image_data['image_path'])) if not "clopping" in raw_data["image_path"][index]]
            coord_data = [[eval(coord_data['start_coordinate'][index])[0],
                            eval(coord_data['start_coordinate'][index])[1],
                            eval(coord_data['end_coordinate'][index])[0],
                            eval(coord_data['end_coordinate'][index])[1]]
                            for index in range(range_data) if not "clopping" in raw_data["image_path"][index]]
            # coord_data = [[coord_data['x_start'][index],
            #                 coord_data['y_start'][index],
            #                 coord_data['x_end'][index],
            #                 coord_data['y_end'][index]]
            #                 for index in range(range_data)]
        else:
            image_data = [image_data['image_path'][index]
                                for index in range(len(image_data['image_path']))]
            coord_data = [[eval(coord_data['start_coordinate'][index])[0],
                            eval(coord_data['start_coordinate'][index])[1],
                            eval(coord_data['end_coordinate'][index])[0],
                            eval(coord_data['end_coordinate'][index])[1]]
                            for index in range(range_data)]
            # coord_data = [[coord_data['x_start'][index],
            #                 coord_data['y_start'][index],
            #                 coord_data['x_end'][index],
            #                 coord_data['y_end'][index]]
            #                 for index in range(range_data)]
        return (image_data, coord_data)

    # dataset loader
    def npz_loader(self, file_name, clop_check):
        npz_image_data = []
        with open(file_name) as file:
            for line in file:
                npz_image_data.append(line.replace("\n", ""))
            image_data = []
            coord_data = []
        for data in npz_image_data:
                if "clopping" in data and clop_check:
                    continue
                tuple_data = []
                binary_data = self.encode(data)/255
                height, width = binary_data.shape
                height_data = np.where(np.any(binary_data == 1, axis=1) == 1)
                way_point = height_data[0][height_data[0]==int(height * 0.8)]
                if len(way_point) == 0:
                    continue
                tuple_data.append(int(np.average(np.where(binary_data[height_data[0][-1]] == 1))) / width)
                tuple_data.append(1.0)
                tuple_data.append(int(np.average(np.where(binary_data[way_point] == 1)[1])) / width)
                tuple_data.append(0.8)
                image_data.append(data.replace(".npz", ".png"))
                coord_data.append(tuple_data)
        return (image_data, coord_data)

    def encode(self, binary_data):
        # load npz-file to image array
        npz_mat = np.load(binary_data)
        return npz_mat['arr_0']

    def setting_graph(self, graph, coordinate_data, bins, title):
        weight = bins[1] - bins[0]
        coordinate_data = coordinate_data * weight
        # graph.title(title)
        graph.set_xlabel(title)
        graph.set_ylabel("error(normalized)")
        graph.set_xlim(-0.6, 0.6)
        graph.set_ylim(0.0, 1.0)
        graph.bar(bins[:-1], coordinate_data, weight, align='edge')
        graph.grid()
        print(bins[:-1])
        print(bins)

if __name__=="__main__":
    object = DataCompare(TARGET_DATASET_PATH, OBJECT_DATASET_PATH)