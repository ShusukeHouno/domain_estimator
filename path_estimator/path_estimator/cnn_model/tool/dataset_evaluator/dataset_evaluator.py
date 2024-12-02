#!/usr/bin/python python3
# encode : utf-8

import sys
import os

sys.path.append("/tf/")

# from settings.setting import setting_tensorflow
from model.predict_model import PredictModel
import tqdm
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from settings import *
import matplotlib.pyplot as plt


class Evaluator:
    # the edge of the bins
    BINS = [
        -0.65,
        -0.55,
        -0.45,
        -0.35,
        -0.25,
        -0.15,
        -0.05,
        0.05,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
    ]

    def __init__(
        self, 
        model, 
        weight_info,
        test_dataset_for_eval,
        result_name,
        debug=False
        ):
        """Evaluate dataset by using trained CNN model.
        By comparing path-label with path estimated with CNN model,
        evaluate validity of the dataset.

        Args:
            model (keras.Sequential): PredictModel.get_model()
            test_dataset_for_eval (str): path-label and img. Is the test_data(or dataset not used to train) recommended as target better than the other source?
        """
        self.model = model
        self.weight_info = weight_info
        self.test_dataset_for_eval = test_dataset_for_eval
        
        self.debug_mode = debug
        
        self.result_name = result_name
        self.result_save_path = "/tf/evaluation_result/" + result_name + "/"
        self.result_dbg_img_path = self.result_save_path + "debug_img/"
        self.result_log_path = self.result_save_path + "log.txt"
        
        os.umask(0)
        os.makedirs(self.result_save_path, mode=0o777, exist_ok=True)
        os.makedirs(self.result_dbg_img_path, mode=0o777, exist_ok=True)
        
        self.csv_loader(test_dataset_for_eval)

        print("Loaded {} images".format(len(self.image_data)))
        
        self.fig = plt.figure(figsize=(10, 5))

    def evaluate(self):
        diff_list = self.calc_path_diff()
        diff_list = np.array(diff_list)

        mean = np.mean(diff_list, axis=0)
        std = np.std(diff_list, axis=0)

        print("mean [sx_m, sy_m, ex_m, ey_m] : {}".format(mean))
        print("std  [sx_d, sy_d, ex_d, ey_d] : {}".format(std))
        
        # Count the data inside 20% error range for respective axis
        error_range_data_num = np.count_nonzero((-0.2 < diff_list) & (diff_list < 0.2), axis=0)
        
        print("Data num inside 20%\ error range [sx_cnt, sy_cnt, ex_cnt, ey_cnt] : {}".format(error_range_data_num))
        
        self.output_log(mean, std, error_range_data_num)
        
        start_x_diff_hist, _ = np.histogram(diff_list[:, :1], bins=Evaluator.BINS, range=(-0.65, 0.65), density=True)
        end_x_diff_hist, _ = np.histogram(diff_list[:, 2:3], bins=Evaluator.BINS, range=(-0.65, 0.65), density=True)
        
        self.draw_graph(self.fig.add_subplot(1, 2, 1), start_x_diff_hist, Evaluator.BINS, "start_x")
        self.draw_graph(self.fig.add_subplot(1, 2, 2), end_x_diff_hist, Evaluator.BINS, "end_x")
        
        plt.savefig(self.result_save_path + "result.pdf", format="pdf", dpi=300)

    def csv_loader(self, file_name: str):
        """Load csv file

        Args:
            file_name (str): target path
        """
        self.image_data = pd.read_csv(file_name, usecols=["image_path"])
        self.coord_data = pd.read_csv(
            file_name, usecols=["start_coordinate", "end_coordinate"]
        )
        self.image_data = [
            self.image_data["image_path"][index]
            for index in range(len(self.image_data["image_path"]))
        ]
        self.coord_data = [
            [
                eval(self.coord_data["start_coordinate"][index])[0],
                eval(self.coord_data["start_coordinate"][index])[1],
                eval(self.coord_data["end_coordinate"][index])[0],
                eval(self.coord_data["end_coordinate"][index])[1],
            ]
            for index in range(len(self.image_data))
        ]

    def calc_path_diff(self) -> list:
        """Calculate the differences between the respective axis the (test)path-label and estimated path.
        
        [ 
            sx_label - sx_est,
            sy_label - sy_est,
            ex_label - ex_est,
            ey_label - ey_est
        ]
        
        Args:

        Returns:
            list: calculated differences
        """
        diff_list = []
        for index, image_path in enumerate(tqdm.tqdm(self.image_data)):
            image = np.array(Image.open(image_path).resize((240, 128))) / 255.0
            image = image[np.newaxis, ...]
            data = self.model.predict(image)
            # path.append(data[0])
            coord = np.array(self.coord_data[index])
            diff_list.append(coord - data[0])
            if self.debug_mode:
                self.draw_dbg_img(image_path, data[0], self.coord_data[index], index)
        return diff_list

    def draw_dbg_img(self, image_path, data, coord, index):
        canvas_image = Image.open(image_path)
        width, height = canvas_image.width, canvas_image.height
        canvas = ImageDraw.Draw(canvas_image)
        canvas.line(
            [(data[0] * width, data[1] * height), (data[2] * width, data[3] * height)],
            fill=(255, 255, 0),
            width=5,
        )
        canvas.line(
            [
                (coord[0] * width, coord[1] * height),
                (coord[2] * width, coord[3] * height),
            ],
            fill=(255, 255, 255),
            width=5,
        )
        index_string = str(index)
        canvas_image.save(self.result_dbg_img_path + "dbg_img_" + index_string.zfill(4) + ".png")

    def draw_graph(self, graph : plt.Axes, target_hist, bins, title):
        class_width = bins[1] - bins[0]
        target_hist *= class_width # target_hist = freq / (class_width * total of freq)
        graph.set_xlabel(title)
        graph.set_ylabel("diff_proportion(normalized)")
        graph.set_xlim(-0.6, 0.6)
        graph.set_ylim(0.0, 0.6)
        graph.bar(bins[:-1], target_hist, class_width, align="edge")
        graph.grid()
    
    def output_log(self, mean, std, data_in_error_range):
        log_file = open(self.result_log_path, "w")
        
        log_file.write("============== EVALUATION LOG ==============\r\n")
        log_file.write("evaluate result name             : {}\r\n".format(self.result_name))
        log_file.write("used CNN weight                  : {}\r\n".format(self.weight_info))
        log_file.write("used test dataset for evaluation : {}\r\n\n".format(self.test_dataset_for_eval))
        log_file.write("-- statistics --\r\n")
        log_file.write(" mean [sx_m, sy_m, ex_m, ey_m] : {}\r\n".format(mean))
        log_file.write(" std  [sx_d, sy_d, ex_d, ey_d] : {}\r\n".format(std))
        log_file.write(" Data num inside 20% error range [sx_cnt, sy_cnt, ex_cnt, ey_cnt] : {}\r\n".format(data_in_error_range))
        
        log_file.close()

if __name__ == "__main__":
    # setting_tensorflow(0)
    model = PredictModel(
        model_name="CCP", 
        weight_path="/tf/weight/fix_sakaki_20230920_r7_l_2_model_name_CCP_epochs100", 
        estimation=True
    )
    
    evaluator = Evaluator(
        model=model.get_model(),
        weight_info=model.get_weight_name(),
        test_dataset_for_eval="/generated_dataset/sakaki_curv_test_ds/generated_train_dataset.csv",
        result_name="sakaki20231005_r7_l2_curv_eval",
        debug=True
    )
    
    # model = PredictModel(
        
    # )
    
    evaluator.evaluate()