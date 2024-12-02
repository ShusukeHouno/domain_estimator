import pandas as pd
import numpy as np
from PIL import Image

class Dataset():
    def __init__(self, data_csv):
        self.image_path, self.label_data = self.analys(data_csv)
    # input data analys and normalization from data list
    def analys(self, csv):
        input_image_path = pd.read_csv(csv, usecols=['image_path'])
        input_data_list = pd.read_csv(csv, usecols=['x_start', 'y_start', 'x_end', 'y_end'], dtype=float)
        data_list = []
        image_path_list = []
        for index, data in enumerate(input_data_list.values):
            # print("input_data(start):" + str(data[0]) + ", " + str(data[1]))
            # print("input_data(end)  :" + str(data[2]) + ", " + str(data[3]))
            image = Image.open(input_image_path['image_path'][index])
            image_path_list.append(input_image_path['image_path'][index])
            width, height = image.size
            if data[0] > 1.0:
                data[0] /= width
                data[2] /= width
                data[1] /= height
                data[3]  /= height
            data_list.append(data)
        print("input data normalize!")
        return image_path_list, data_list
    def getData(self):
        return self.image_path, self.label_data
    def updateData(self, image_path, label_data):
        self.image_path = image_path
        self.label_data = label_data
        
if __name__ == '__main__':
    data_set = Dataset("../sakaki_data_train.csv")
