import sys

sys.path.append("/tf/")

from model.predict_model import PredictModel
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import os

import cv2
import tqdm


class ModelChecker:
    def __init__(
        self,
        target_model_name: str,
        target_ds_name: str,
        weight: str,
        is_only_point_dataset: bool,
        max_branch_num: int,
    ):
        model = PredictModel(
            model_name=target_model_name, weight_path=weight, estimation=True
        )

        self.model = model.get_model()
        self.max_branch_num = max_branch_num
        self.target_ds_file = (
            "/generated_dataset/" + target_ds_name + "/generated_test_dataset.csv"
        )
        self.result_dir = "/tf/check_result/model/" + target_ds_name + "/"
        self.result_img_dir = self.result_dir + "img/"
        self.is_only_point_dataset = is_only_point_dataset

        os.umask(0)
        os.makedirs(self.result_dir, mode=0o777, exist_ok=True)
        os.makedirs(self.result_img_dir, mode=0o777, exist_ok=True)

        self.img_header = ["image_path"]
        self.coords_header = []
        if self.max_branch_num == 1:
            if not self.is_only_point_dataset:
                self.coords_header.append("start_coordinate")
            self.coords_header.append("end_coordinate")
        else:
            for idx in range(self.max_branch_num):
                if not self.is_only_point_dataset:
                    self.coords_header.append("start_coordinate_" + str(idx))
                self.coords_header.append("end_coordinate_" + str(idx))

        # 原則としてvalidation dataのみチェック
        self.img_list, self.label_list = self._load()
        self.data_len = len(self.img_list)

    def _load(self):
        image_path_list = pd.read_csv(self.target_ds_file, usecols=self.img_header)
        data_path_list = pd.read_csv(self.target_ds_file, usecols=self.coords_header)

        # target image
        image_list = [
            image_path_list["image_path"][idx] for idx in range(len(image_path_list))
        ]

        # label
        label_list = []
        for idx in range(len(image_path_list["image_path"])):
            label_elem = []
            for head in self.coords_header:
                label_elem.append(float(eval(data_path_list[head][idx])[0]))
                label_elem.append(float(eval(data_path_list[head][idx])[1]))
            label_list.append(label_elem)

        return image_list, label_list

    def _normalize_img(self, img: cv2.Mat):
        ret_img = cv2.resize(img, (240, 128))
        ret_img = np.array(ret_img, np.float32)
        ret_img /= 255.0
        return ret_img

    def _get_input_img(self, img_path: str):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret = self._normalize_img(img)
        ret = ret.reshape(1, 128, 240, 3)
        return ret

    def _resize_coordinate(self, coords_list: list, w: int, h: int):
        ret = []
        diff = 2 if self.is_only_point_dataset else 4
        for idx in range(self.max_branch_num):
            ret.append(coords_list[idx * diff + 0] * w)
            ret.append(coords_list[idx * diff + 1] * h)
            if not self.is_only_point_dataset:
                ret.append(coords_list[idx * diff + 2] * w)
                ret.append(coords_list[idx * diff + 3] * h)
        return ret

    def _writeout_line(self, coords_list: list, canvas: ImageDraw, color: tuple):
        for idx in range(self.max_branch_num):
            canvas.ellipse(
                (
                    coords_list[idx * 4 + 0] - 3,
                    coords_list[idx * 4 + 1] - 3,
                    coords_list[idx * 4 + 0] + 3,
                    coords_list[idx * 4 + 1] + 3,
                ),
                fill=color,
            )
            canvas.ellipse(
                (
                    coords_list[idx * 4 + 2] - 3,
                    coords_list[idx * 4 + 3] - 3,
                    coords_list[idx * 4 + 2] + 3,
                    coords_list[idx * 4 + 3] + 3,
                ),
                fill=color,
            )
            canvas.line(
                (
                    coords_list[idx * 4 + 0],
                    coords_list[idx * 4 + 1],
                    coords_list[idx * 4 + 2],
                    coords_list[idx * 4 + 3],
                ),
                fill=color,
                width=4,
            )

    def _writeout_point(self, coords_list: list, canvas: ImageDraw, color: tuple):
        for idx in range(self.max_branch_num):
            canvas.ellipse(
                (
                    coords_list[idx * 2 + 0] - 5,
                    coords_list[idx * 2 + 1] - 5,
                    coords_list[idx * 2 + 0] + 5,
                    coords_list[idx * 2 + 1] + 5,
                ),
                fill=color,
            )

    def check_estimation(self):
        for idx in tqdm.tqdm(range(self.data_len)):
            target_img = self.img_list[idx]
            label_coords = self.label_list[idx]
            img_for_draw = Image.open(target_img)
            img_canvas = ImageDraw.Draw(img_for_draw)
            w, h = img_for_draw.size

            input_img = self._get_input_img(target_img)

            estimate_coords = self.model.predict(input_img)

            resized_estimate_coords = self._resize_coordinate(estimate_coords[0], w, h)

            resized_label_coords = self._resize_coordinate(label_coords, w, h)

            if not self.is_only_point_dataset:
                self._writeout_line(resized_estimate_coords, img_canvas, (0, 255, 0))
                self._writeout_line(resized_label_coords, img_canvas, (255, 0, 0))
            else:
                self._writeout_point(resized_estimate_coords, img_canvas, (0, 255, 0))
                self._writeout_point(resized_label_coords, img_canvas, (255, 0, 0))

            img_for_draw.save(self.result_img_dir + "d" + str(idx) + ".png")


if __name__ == "__main__":
    checker = ModelChecker(
        target_model_name="CCP_JCT",
        target_ds_name="sakaki_jct_point_ds202311122337",
        weight="/tf/weight/TEST_JCT_POINT_CCP_JCT_epc100",
        is_only_point_dataset=True,
        max_branch_num=1,
    )

    checker.check_estimation()
