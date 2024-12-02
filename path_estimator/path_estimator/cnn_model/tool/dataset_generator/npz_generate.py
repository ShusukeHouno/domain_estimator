#!/usr/bin/bash python
#coding:utf-8

import argparse
import numpy as np
import glob
import tqdm
from PIL import Image, ImageDraw

class generater():
    def __init__(self, path, debug, replace_path, clopping):
        # self.path_list = glob.glob(path+"*.npz")
        # path list
        self.npz_path_list = self.get_pathlist(path)
        self.img_path_list = [path.replace('.npz', '.png') for path in self.npz_path_list]
        if debug:
            self.debug_imagedraw(replace_path)
        if clopping:
            self.clop_npz("/sakaki_202212", "/clopping_data")

    def get_pathlist(self, path):
        with open(path, 'r') as file:
            path_list = [s.strip() for s in file.readlines()]
        return path_list

    def clop_npz(self, target_path, change_path):
        data = []
        for index, path in enumerate(self.img_path_list):
            image = Image.open(path)
            numpy_image = np.load(path.replace(".png", ".npz"))
            width, height = image.width, image.height
            move_window = [int(width * 0.1), int(height * 0.1)]
            window_size = [int(width * 0.8), int(height * 0.8)]
            numpy_data = numpy_image['arr_0']
            data.append(path.replace(".png", ".npz\n"))
            for roop_num in tqdm.tqdm(range(9), leave=False):
                vertical_num, horizon_num = divmod(roop_num, 3)
                crop_image = image.crop((move_window[0] * horizon_num, move_window[1] * vertical_num, \
                            window_size[0] + move_window[0] * horizon_num, window_size[1] + move_window[1] * vertical_num))
                crop_image.save(path.replace(target_path, target_path + change_path)
                                .replace(".png", "_" + str(roop_num) + ".png"))
                crop_image_numpy = numpy_data[move_window[1] * vertical_num : window_size[1] + move_window[1] * vertical_num,
                                              move_window[0] * horizon_num : window_size[0] + move_window[0] * horizon_num]
                np.savez(path.replace(target_path, target_path + change_path)
                                .replace(".png", "_" + str(roop_num)), crop_image_numpy)
                data.append(path.replace(target_path, target_path + change_path)
                                .replace(".png", "_" + str(roop_num) + ".npz\n"))
                
        with open("/create_dataset/sakaki_202212/dataset_npz/train_aug.txt", "w") as file:
            for data_ in data:  file.write(data_)
    def debug_imagedraw(self, replace):
        for img_path, npz_path in zip(self.img_path_list, self.npz_path_list):
            print([img_path, npz_path])
            image = Image.open(img_path).resize((960, 540))
            npz_data = np.load(npz_path)
            point_data = np.where(npz_data['arr_0']!=0)
            coordinate_list = ([(point_data[1][index], point_data[0][index])
                                for index in range(len(point_data[0]))])
            canvas = ImageDraw.Draw(image)
            canvas.point(coordinate_list, fill=(255,255,0))
            image.save(img_path.replace(replace, 'debug'))

def arg_manage():
    parser = argparse.ArgumentParser(description="Formatting the dataset")
    parser.add_argument('path', type=str, help='Specify the directory where npz is located')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-s', '--save_name')
    argument = parser.parse_args()
    return argument

def image_draw(image_path):
    image = Image.open(image_path.replace("/path/train/", "/image/").replace(".npz", ".jpg")).resize((960, 540))
    canvas = ImageDraw.Draw(image)
    npz_file_data = np.load(image_path)
    point_data_open = np.where(npz_file_data['arr_0'] != 0)
    point_data = ([(point_data_open[1][index], point_data_open[0][index]) for index in range(len(point_data_open[0]))])
    canvas.point(point_data, fill=(255, 255, 0))
    image.save(image_path.replace("/path/train/", "/dataset_view/").replace(".npz", ".png"))


if __name__=='__main__':
    arg = arg_manage()
    dataset_generater = generater(arg.path, arg.debug, arg.save_name, True)