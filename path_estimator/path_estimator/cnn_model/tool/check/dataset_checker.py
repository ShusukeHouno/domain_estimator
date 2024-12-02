#!/usr/bin/env python

import csv
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import os
import tqdm

class DatasetChecker:
    def _approx_zero(self, value):
        return value <= 1e-2

    def __init__(self, generated_ds_name: str, max_branch_num=2):
        self.train_ds_dir = (
            "/generated_dataset/" + generated_ds_name + "/generated_train_dataset.csv"
        )
        self.validation_ds_dir = (
            "/generated_dataset/" + generated_ds_name + "/generated_test_dataset.csv"
        )
        self.result_dir = "/tf/check_result/ds/" + generated_ds_name + "/"
        self.result_img_dir = self.result_dir + "img/"
        self.max_branch_num = max_branch_num
        self.img_header = ["image_path"]
        self.coords_header = []

        if self.max_branch_num == 1:
            self.coords_header.append("start_coordinate")
            self.coords_header.append("end_coordinate")
        else:
            for idx in range(self.max_branch_num):
                self.coords_header.append("start_coordinate_" + str(idx))
                self.coords_header.append("end_coordinate_" + str(idx))

        os.umask(0)
        os.makedirs(self.result_dir, mode=0o777, exist_ok=True)
        os.makedirs(self.result_img_dir, mode=0o777, exist_ok=True)

        self.train_img_list, self.train_path_list = self._load(self.train_ds_dir)
        self.validation_img_list, self.validation_path_list = self._load(
            self.validation_ds_dir
        )

    def _load(self, ds_dir):
        image_path_list = pd.read_csv(ds_dir, usecols=self.img_header)
        data_path_list = pd.read_csv(ds_dir, usecols=self.coords_header)

        # target image
        image_list = [
            image_path_list["image_path"][idx] for idx in range(len(image_path_list))
        ]

        # label
        label_list = []
        for idx in range(len(image_path_list["image_path"])):
            label_elem = []
            for head in self.coords_header:
                label_elem.append(eval(data_path_list[head][idx])[0])
                label_elem.append(eval(data_path_list[head][idx])[1])
            label_list.append(label_elem)

        return image_list, label_list

    def _get_correct_multi_path(self, path_list: list):
        multi_path_idx = []
        for path_idx in range(len(path_list)):
            label = path_list[path_idx]
            pre_coords = []
            for idx in range(self.max_branch_num):
                tmp_start_coords = [
                    label[idx * 4 + 0],
                    label[idx * 4 + 1],
                ]
                tmp_end_coords = [
                    label[idx * 4 + 2],
                    label[idx * 4 + 3],
                ]

                if pre_coords:
                    pre_start_np_array = np.array(pre_coords[0:2])
                    pre_end_np_array = np.array([pre_coords[2:4]])

                    tmp_start_np_array = np.array(tmp_start_coords)
                    tmp_end_np_array = np.array(tmp_end_coords)

                    start_diff_norm = np.linalg.norm(
                        tmp_start_np_array - pre_start_np_array
                    )
                    end_diff_norm = np.linalg.norm(tmp_end_np_array - pre_end_np_array)

                    # print(start_diff_norm)
                    if not (
                        self._approx_zero(start_diff_norm)
                        and self._approx_zero(end_diff_norm)
                    ):
                        multi_path_idx.append(path_idx)
                        continue

                pre_coords = tmp_start_coords + tmp_end_coords

        return multi_path_idx

    def _writeout_line(self, img_list, path_list, idx_list, base_name):
        print(len(idx_list))
        for idx in tqdm.tqdm(idx_list):
            # pass
            target_img = Image.open(img_list[idx])
            target_canvas = ImageDraw.Draw(target_img)
            
            w, h = target_img.size
            coords = path_list[idx]
            # print(len(coords))

            for path_idx in range(self.max_branch_num):
                target_canvas.ellipse(
                    (
                        float(coords[path_idx * 4 + 0]*w) - 3,
                        float(coords[path_idx * 4 + 1]*h) - 3,
                        float(coords[path_idx * 4 + 0]*w) + 3,
                        float(coords[path_idx * 4 + 1]*h) + 3,
                    ),
                    fill=(255, 0, 0),
                )
                target_canvas.ellipse(
                    (
                        float(coords[path_idx * 4 + 2]*w) - 3,
                        float(coords[path_idx * 4 + 3]*h) - 3,
                        float(coords[path_idx * 4 + 2]*w) + 3,
                        float(coords[path_idx * 4 + 3]*h) + 3,
                    ),
                    fill=(255, 0, 0),
                )
                target_canvas.line(
                    (
                        float(coords[path_idx * 4 + 0]*w),
                        float(coords[path_idx * 4 + 1]*h),
                        float(coords[path_idx * 4 + 2]*w),
                        float(coords[path_idx * 4 + 3]*h),
                    ),
                    fill=(255, 0, 0),
                    width=4,
                )
            target_img.save(self.result_img_dir + base_name + str(idx) + ".png")

    def check_multi_path(self):
        """Check how many labels exist for which more than 2 lines are correctly detected."""
        if self.max_branch_num == 1:
            return False

        train_multi_path_idx = self._get_correct_multi_path(self.train_path_list)
        validation_multi_path_idx = self._get_correct_multi_path(
            self.validation_path_list
        )

        print("train multi path cnt : {}".format(len(train_multi_path_idx)))
        print(
            "train multi path ratio : {}".format(
                len(train_multi_path_idx) / len(self.train_path_list)
            )
        )
        print("validation multi path cnt : {}".format(len(validation_multi_path_idx)))
        print(
            "validation multi path ratio : {}".format(
                len(validation_multi_path_idx) / len(self.validation_path_list)
            )
        )

        self._writeout_line(
            self.train_img_list, self.train_path_list, train_multi_path_idx, "train"
        )
        self._writeout_line(
            self.validation_img_list,
            self.validation_path_list,
            validation_multi_path_idx,
            "validation",
        )


if __name__ == "__main__":
    checker = DatasetChecker(
        generated_ds_name="sakaki_r9_l15_multi202311052242", max_branch_num=2
    )

    checker.check_multi_path()
