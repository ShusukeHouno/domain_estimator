#!/usr/bin/bin python
# coding: utf-8

import argparse
import csv
import tqdm
import numpy as np
import os
import random
import pandas as pd
import datetime
from PIL import Image, ImageDraw

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")


class DatasetGenerator:
    def _is_the_last_index(self, idx: int, target_list: list) -> bool:
        return idx == (len(target_list) - 1)

    def _is_valid_coords(self, coords_list: int) -> bool:
        return len(coords_list) == 2

    def _is_before_junction(self, start_idx_list: list) -> bool:
        sorted_idx_list = sorted(start_idx_list)
        return sorted_idx_list[0] > 0

    def _is_img_in_range(self, l_w, r_w, u_h, d_h, coords) -> bool:
        return (l_w <= float(coords[0]) <= r_w) and (u_h <= float(coords[1]) <= d_h)

    def _normalize_coordinate(self, w, h, coords) -> bool:
        ret = [coords[0] / w, coords[1] / h]
        return ret

    def __init__(
        self,
        raw_dataset_dir_list: list,
        dataset_save_base_name: str,
        dataset_train_ratio=0.7,
        is_only_point_dataset=False,
        is_const_point_labeling=False,
        is_img_based_labeling=False,
        dataset_distance_threshold=1.5,
        is_multi_path_labeling=False,
        max_branch_num=2,
        debug=False,
    ):
        """_summary_

        Args:
            parent (str): wakaran
            raw_dataset_path_list (list[str]): raw_dataset dir list, the input is expected to be /raw_dataset/{dataset_name}
        """
        self.raw_dataset_list = raw_dataset_dir_list
        self.dataset_save_name = dataset_save_base_name + datetime.datetime.now(
            JST
        ).strftime("%Y%m%d%H%M")
        self.dataset_save_path = "/generated_dataset/" + self.dataset_save_name + "/"
        self.dataset_cropped_img_path = self.dataset_save_path + "cropped_data/"
        self.dataset_train_csv_path = (
            self.dataset_save_path + "generated_train_dataset.csv"
        )
        self.dataset_validation_csv_path = (
            self.dataset_save_path + "generated_test_dataset.csv"
        )
        self.dataset_log_path = self.dataset_save_path + "log.txt"
        self.dataset_debug_img_path = self.dataset_save_path + "debug_img/"
        self.dataset_debug_point_img_path = self.dataset_save_path + "debug_point/"
        os.umask(0)
        os.makedirs(self.dataset_save_path, mode=0o777, exist_ok=True)
        os.makedirs(self.dataset_cropped_img_path, mode=0o777, exist_ok=True)
        os.makedirs(self.dataset_debug_img_path, 0o777, exist_ok=True)
        os.makedirs(self.dataset_debug_point_img_path, 0o777, exist_ok=True)

        self.dataset_pair_list = []  # [img_path, coords_list[]]
        self.dataset_validation_pair_list = []
        self.dataset_path_idx = 0
        self.dataset_train_ratio = dataset_train_ratio
        self.train_dataset_size = 0
        self.test_dataset_size = 0

        self.is_const_point_labeling = is_const_point_labeling

        self.is_img_based_labeling = is_img_based_labeling
        self.REAL_DISTANCE_THRESHOLD = dataset_distance_threshold

        self.is_multi_path_labeling = is_multi_path_labeling
        self.MAX_BRANCH_NUM = max_branch_num

        self.is_only_point_dataset = is_only_point_dataset
        self.cam_internal_param = [915.072, 0, 650.107, 0, 914.836, 357.393]

    def generate(self):
        """Generate path-label"""
        # generate dataset from all raw_dataset
        for dataset_path in self.raw_dataset_list:
            # generate loading path
            self.raw_dataset_csv_path = dataset_path + "/raw_dataset.csv"
            self.cam_internal_param_csv_path = dataset_path + "/camera_param.csv"

            # load metadata and dataset csv
            self.load_cam_internal_param()
            self.load_csv_data()

            # get all image_path from data_list
            self.image_path_list = [path_list[0] for path_list in self.data_list]

            # open first image, and get width and height
            image_data = Image.open(self.image_path_list[0])
            self.width, self.height = image_data.size
            
            if self.is_const_point_labeling:
                self.calc_junction_point(True)
            else:
                if self.is_multi_path_labeling:
                    self.calc_multiple_passable_line_segment(True)
                else:
                    self.calc_passable_line_segment(True)
            # self.test_calc_passable_line_segment(True)
            self.dataset_path_idx += 1

        if self.is_only_point_dataset and not self.is_const_point_labeling:
            self.convert_passable_multi_point(True)
        self.separate_train_test()
        self.write_out_dataset()
        self.output_log()
        # self.write_out_debug()

    def test_generate(self):
        for dataset_path in self.raw_dataset_list:
            self.raw_dataset_csv_path = dataset_path + "/raw_dataset.csv"
            self.cam_internal_param_csv_path = dataset_path + "/camera_param.csv"

            # load metadata and dataset csv
            self.load_cam_internal_param()
            self.load_csv_data()

            # get all image_path from data_list
            self.image_path_list = [path_list[0] for path_list in self.data_list]

            # open first image, and get width and height
            image_data = Image.open(self.image_path_list[0])
            self.width, self.height = image_data.size

            # self.calc_multiple_passable_line_segment(False, False)
            self.calc_junction_point(True)
            # self.test_calc_passable_line_segment(True)
            self.dataset_path_idx += 1

        # self.test_calc_passable_multi_point(False)

    def load_csv_data(self):
        """
        open csv and load data

        self.data_list : ['path', [x0, y0], ... , [xn, yn]]
        """
        self.data_list = []
        print(self.raw_dataset_csv_path)
        with open(self.raw_dataset_csv_path, "r", encoding="utf-8") as file:
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
                    self.data_list.append(row_data)

    def load_cam_internal_param(self):
        """load camera internal parameter from internal param.csv"""
        print("loading internal param")
        data_frame = pd.read_csv(
            self.cam_internal_param_csv_path, names=["fx", "cx", "fy", "cy"]
        )
        self.cam_internal_param = [
            float(data_frame["fx"][1]),
            0.0,
            float(data_frame["cx"][1]),
            0.0,
            float(data_frame["fy"][1]),
            float(data_frame["cy"][1]),
        ]
        print(self.cam_internal_param)

    def output_log(self):
        """Output generating logs"""
        log_file = open(self.dataset_log_path, "w")

        log_file.write("============== TRAIN/TEST DATASET GEN LOG ==============\r\n")
        log_file.write("generated dataset name : {}\r\n".format(self.dataset_save_name))
        log_file.write("raw_dataset name : \r\n")
        for raw_ds_name in self.raw_dataset_list:
            log_file.write(" - {}\r\n".format(raw_ds_name))

        if self.is_only_point_dataset:
            log_file.write("dataset format : point only\r\n")
        else:
            log_file.write("dataset format : path format\r\n")

        if self.is_const_point_labeling:
            log_file.write("dataset labeling method : const point\r\n")
        else:
            if self.is_img_based_labeling:
                log_file.write("dataset labeling method : image range based\r\n")
            else:
                log_file.write("dataset labeling method : real-distance based\r\n")
                log_file.write(
                    " - distance threshold : {} [m]\r\n".format(
                        self.REAL_DISTANCE_THRESHOLD
                    )
                )

        if self.is_multi_path_labeling:
            log_file.write("multiple path labeling configuration\r\n")
            log_file.write(" - max branch num : {}".format(self.MAX_BRANCH_NUM))

        log_file.write(
            "total dataset size : {} pcs\r\n".format(
                self.train_dataset_size + self.test_dataset_size
            )
        )
        log_file.write(
            "train dataset size : {} pcs ({}%)\r\n".format(
                self.train_dataset_size, self.dataset_train_ratio * 100.0
            )
        )
        log_file.write(
            "test  dataset size : {} pcs ({}%)\r\n".format(
                self.test_dataset_size, (1.0 - self.dataset_train_ratio) * 100.0
            )
        )
        log_file.close()

    def resave_img(self, original_path, new_path):
        """Resave image

        Args:
            original_path (_type_): _description_
            new_path (_type_): _description_
        """
        img = Image.open(original_path)
        img.save(new_path)

    def test_calc_passable_line_segment(self, save_flag):
        print(len(self.data_list))
        for idx in tqdm.tqdm(range(0, len(self.data_list) - 1)):
            coords_list = self.gen_img_based_edge_points(
                self.data_list[idx],
                0,
                self.width,
                self.height,
                # 2.0
            )

            if len(coords_list) == 2:
                uncropped_coords = [
                    [coords_list[0][0] / self.width, 1.0],
                    [coords_list[1][0] / self.width, coords_list[1][1] / self.height],
                ]
                self.draw_image(idx, coords_list, self.dataset_debug_img_path)

    def test_calc_passable_multi_point(self, save_flag):
        for idx in range(len(self.dataset_pair_list)):
            ds_pair = [self.dataset_pair_list[idx][0]]
            for path_idx in range(0, self.MAX_BRANCH_NUM * 2, 2):
                start_point = self.dataset_pair_list[idx][path_idx + 0 + 1]
                end_point = self.dataset_pair_list[idx][path_idx + 1 + 1]
                ds_pair.append(end_point)
            self.dataset_pair_list[idx] = ds_pair

    def convert_passable_multi_point(self, save_flag):
        for idx in range(len(self.dataset_pair_list)):
            ds_pair = [self.dataset_pair_list[idx][0]]
            for path_idx in range(0, self.MAX_BRANCH_NUM * 2, 2):
                start_point = self.dataset_pair_list[idx][path_idx + 0 + 1]
                end_point = self.dataset_pair_list[idx][path_idx + 1 + 1]
                ds_pair.append(end_point)
            self.dataset_pair_list[idx] = ds_pair
        # for idx in range(len(self.dataset_pair_list)):
        #     ds_pair = self.dataset_pair_list[idx]
        #     # print(ds_pair)
        #     img = Image.open(ds_pair[0])
        #     canvas = ImageDraw.Draw(img)

        #     for path_idx in range(self.MAX_BRANCH_NUM):
        #         canvas.ellipse(
        #             (
        #                 ds_pair[path_idx+1][0] * img.width - 3,
        #                 ds_pair[path_idx+1][1] * img.height - 3,
        #                 ds_pair[path_idx+1][0] * img.width + 3,
        #                 ds_pair[path_idx+1][1] * img.height + 3,
        #             ),
        #             fill=(255, 0, 0),
        #             width=5
        #         )

        #     img.save(
        #         self.dataset_debug_img_path
        #         + "debug_"
        #         + str(idx)
        #         + ".png"
        #     )

    def calc_junction_point(self, save_flag):
        sample_img = Image.open(self.data_list[0][0])
        w, h = sample_img.size

        for index in tqdm.tqdm(range(0, len(self.data_list) - 1)):
            target_data = self.data_list[index]
            img_path = target_data[0]
            jct_idx = target_data[1][0]
            path_point_len = len(target_data) - 3

            if jct_idx < 0:
                continue

            if path_point_len <= jct_idx:
                continue

            coords = [
                float(target_data[jct_idx + 3][0]),
                float(target_data[jct_idx + 3][1]),
            ]

            if self._is_img_in_range(0, w, 0, h, coords):
                normalized_coords = self._normalize_coordinate(w, h, coords)
                self.dataset_pair_list.append([img_path, normalized_coords])
                if save_flag:
                    self.draw_single_point(
                        img_path,
                        coords,
                        self.dataset_debug_img_path
                        + "debug_"
                        + str(index)
                        + "_"
                        + str(self.dataset_path_idx)
                        + ".png",
                    )

            self.single_point_crop_aug(img_path, coords, index, save_flag)

    def calc_passable_line_segment(self, save_flag):
        """Calc path-label
        tqdm : progress bar

        Args:
            save_flag (bool): what?
            path_name (str): /{parent}/{path_name}/debug_ (/dataset/sakaki_trial1/debug_)
        """
        for index in tqdm.tqdm(range(0, len(self.data_list) - 1)):
            target_data = self.data_list[index]
            coordinate_list, write_data = [], []
            if self.is_img_based_labeling:
                coordinate_list = self.gen_img_based_edge_points(
                    target_data, 0, self.width, self.height
                )
            else:
                coordinate_list = self.gen_real_world_based_edge_points(
                    target_data,
                    0,
                    self.width,
                    self.height,
                    self.REAL_DISTANCE_THRESHOLD,
                )

            # for saving image
            if len(coordinate_list) == 2:
                # the path data is represented by percentage
                if self.is_img_based_labeling:
                    uncropped_coords = [
                        [
                            coordinate_list[0][0] / self.width,
                            1.0,
                        ],
                        [coordinate_list[1][0] / self.width, 0.8],
                    ]
                else:
                    uncropped_coords = [
                        [coordinate_list[0][0] / self.width, 1.0],
                        [
                            coordinate_list[1][0] / self.width,
                            coordinate_list[1][1] / self.height,
                        ],
                    ]
                self.dataset_pair_list.append(
                    [target_data[0], uncropped_coords[0], uncropped_coords[1]]
                )
                if save_flag:
                    self.draw_image(index, coordinate_list, self.dataset_debug_img_path)

            if self.is_img_based_labeling:
                self.crop_augmentation(
                    self.image_path_list[index], target_data, save_flag, index
                )
            else:
                self.real_world_based_crop_aug(
                    self.image_path_list[index],
                    target_data,
                    save_flag,
                    index,
                    self.REAL_DISTANCE_THRESHOLD,
                )

    def calc_multiple_passable_line_segment(self, is_debug, is_aug=True):
        """Calc multiple path-label

        Args:
            is_debug (bool): draw debug img
        """
        for index in tqdm.tqdm(range(0, len(self.data_list) - 1)):
            # index <- camera point
            target_data = self.data_list[index]
            coords_list = []
            coords_list = self.gen_multiple_path_edge_points(
                target_data,
                0,
                self.width,
                self.height,
                self.REAL_DISTANCE_THRESHOLD,
            )

            dataset_pair = [target_data[0]]
            norm_coords = []
            for coordinate in coords_list:
                norm_coords = [
                    [coordinate[0][0] / self.width, 1.0],
                    [coordinate[1][0] / self.width, coordinate[1][1] / self.height],
                ]
                dataset_pair += norm_coords

            if coords_list:
                # padding
                for _ in range(0, self.MAX_BRANCH_NUM - len(coords_list)):
                    dataset_pair += norm_coords  # padding with last coords

                # append to the dataset pair list
                self.dataset_pair_list.append(dataset_pair)

            if is_debug:
                self.draw_multi_line(index, coords_list, self.dataset_debug_img_path)

            if is_aug:
                self.multi_path_crop_aug(
                    self.image_path_list[index],
                    target_data,
                    is_debug,
                    index,
                    self.REAL_DISTANCE_THRESHOLD,
                )

    def gen_img_based_edge_points(self, target_data, left_width, right_width, height):
        """_summary_
        Calcurate line, representing the path

        Args:
            target_data (dataset?): ['img_path', [path(x,y)]...]
            left_width (int): range left? default to 0
            right_width (int): range right? default to width
            height (int): image hight

        Returns:
            list: [[start_x, start_y], [end_x, end_y]]
        """
        coordinate_list = []

        under_line_height = height
        upper_line_height = height * 0.8

        self.debug_point = []

        for idx in range(2, len(target_data)):
            pr_point = [target_data[idx - 1][0], target_data[idx - 1][1]]
            cr_point = [target_data[idx][0], target_data[idx][1]]

            if len(self.debug_point) > 0:
                self.debug_point.append(cr_point)

            if (
                pr_point[1] > under_line_height  # check the prev point is out of image
                and cr_point[1]
                <= under_line_height  # check the current point is in image
                and left_width < pr_point[0] < right_width
                and left_width < cr_point[0] < right_width
            ):
                # print("found")
                # calc intersection of bottom line and pr-cr line
                point_x = self.calc_width_point([pr_point, cr_point], under_line_height)
                coordinate_list.append([int(point_x), under_line_height])
                self.debug_point.append([int(point_x), under_line_height])

            if (
                pr_point[1] > upper_line_height
                and cr_point[1] <= upper_line_height
                and left_width < pr_point[0] < right_width
                and left_width < cr_point[0] < right_width
            ):
                point_x = self.calc_width_point([pr_point, cr_point], upper_line_height)
                coordinate_list.append([int(point_x), int(upper_line_height)])

        return coordinate_list

    def gen_multiple_path_edge_points(
        self,
        target_data: list,
        left_width: int,
        right_width: int,
        height: int,
        threshold_distance=1.5,
    ):
        """Calculate multiple path edge points
        Args:
            target_data (list): ["img", [branch_start_idx], [path_point_0], ...]
            left_width (int): _description_
            right_width (int): _description_
            height (int): _description_
            threshold_distance (float, optional): _description_. Defaults to 1.5.
        """
        ret_coords_list = []
        branch_start_idx_list = target_data[1]

        all_point_list = [target_data[i] for i in range(2, len(target_data))]

        if self._is_before_junction(branch_start_idx_list):
            # current index(cam_point) is before the junction point
            base_point_list = [
                all_point_list[i] for i in range(0, branch_start_idx_list[0])
            ]  # base point <- start to junction
            for idx, branch_start_idx in enumerate(branch_start_idx_list):
                branch_end_idx = len(all_point_list)  # for last branch
                if not self._is_the_last_index(idx, branch_start_idx_list):
                    branch_end_idx = branch_start_idx_list[
                        idx + 1
                    ]  # 次のbranch start pointの一個まえまでがこのbranchの区間
                # to use gen_edge_points, add empty element as first
                formatted_target_data = (
                    [""]
                    + base_point_list
                    + [
                        all_point_list[i]
                        for i in range(branch_start_idx, branch_end_idx)
                    ]
                )

                coords_list = self.gen_real_world_based_edge_points(
                    formatted_target_data,
                    left_width,
                    right_width,
                    height,
                    threshold_distance,
                )

                if not self._is_valid_coords(coords_list):
                    continue

                ret_coords_list.append(coords_list)

        else:
            # branchごとにラベリング
            for idx, branch_start_idx in enumerate(branch_start_idx_list):
                start_idx = 0  # この場合基本は始点から
                branch_end_idx = len(all_point_list)

                if not self._is_the_last_index(idx, branch_start_idx_list):
                    branch_end_idx = (
                        branch_start_idx_list[idx + 1]
                        if branch_start_idx_list[idx + 1] <= len(all_point_list)
                        else len(all_point_list)
                    )

                # branchの区間が存在するか？
                if branch_end_idx <= 0:
                    continue

                # 明確な始点が存在する？
                if branch_start_idx > 0:
                    start_idx = branch_start_idx

                formatted_target_data = [""] + [
                    all_point_list[i] for i in range(start_idx, branch_end_idx)
                ]

                coords_list = self.gen_real_world_based_edge_points(
                    formatted_target_data,
                    left_width,
                    right_width,
                    height,
                    threshold_distance,
                )

                if not self._is_valid_coords(coords_list):
                    continue

                ret_coords_list.append(coords_list)

        # print("ret coords len {}".format(len(ret_coords_list)))
        return ret_coords_list

    def gen_real_world_based_edge_points(
        self, target_data: list, left_width, right_width, height, threshold_distance=1.5
    ):
        """Generate path-label based on real world distance

        Args:
            target_data (list): row data of input csv (['img_path', point_0, point_1, ...]). Input csv must contain real-world z-axis value for all point data.
            threshold_distance (float, optional): Threshould length of path-label. Defaults to 1.5 [m].

        Returns:
            list: coordinate list (['start_point', 'end_point'])
        """
        coords_list = []

        under_line_height = height
        total_distance = 0.0
        tmp_total = 0
        for idx in range(2, len(target_data)):
            pr_data = target_data[idx - 1]
            cr_data = target_data[idx]
            pr_img_point = [pr_data[0], pr_data[1]]
            cr_img_point = [cr_data[0], cr_data[1]]
            pr_world_point = self.img_to_world_point(
                pr_img_point, self.cam_internal_param, pr_data[2]
            )
            cr_world_point = self.img_to_world_point(
                cr_img_point, self.cam_internal_param, cr_data[2]
            )
            if (
                pr_img_point[1] > under_line_height
                and cr_img_point[1]
                <= under_line_height  # check the current point is in image
                and left_width < pr_img_point[0] < right_width
                and left_width < cr_img_point[0] < right_width
            ):
                point_x = self.calc_width_point(
                    [pr_img_point, cr_img_point], under_line_height
                )
                coords_list.append([int(point_x), under_line_height])
                pr_world_point = self.img_to_world_point(
                    coords_list[0], self.cam_internal_param
                )
                # print("found pre point")

            if len(coords_list) > 0:
                distance = np.linalg.norm(cr_world_point - pr_world_point)
                tmp_total = total_distance + distance
                if tmp_total < threshold_distance:
                    total_distance += distance
                elif tmp_total == threshold_distance:
                    # img point check
                    # 今の所は先行研究に合わせて範囲外なら消す
                    if left_width < cr_img_point[0] < right_width:
                        coords_list.append([int(cr_img_point[0]), int(cr_img_point[1])])
                    break
                else:
                    remain_length = threshold_distance - total_distance
                    middle_point = (
                        (cr_world_point - pr_world_point) / distance
                    ) * remain_length + pr_world_point
                    # print(pr_world_point - pr_world_point_truth)
                    # print(middle_point)
                    middle_img_point = self.world_to_img_point(
                        [middle_point[0], middle_point[1], middle_point[2]],
                        self.cam_internal_param,
                        middle_point[1],
                    )
                    if left_width < middle_img_point[0] < right_width:
                        coords_list.append(
                            [int(middle_img_point[0]), int(middle_img_point[1])]
                        )
                    break
        return coords_list

    def img_to_world_point(
        self, img_point: list, cam_internal_param: list, cam_z_height=1.0
    ):
        """Convert image point to real world 3d coordinate.

        Args:
            img_point (list): [u,v]
            cam_internal_param (list): [fx,cx,fy,cy]
            cam_z_height (float, optional): camera height. Defaults to 1.0.

        Returns:
            list: coordinate
        """
        c_offset = np.array(
            [cam_internal_param[2], cam_internal_param[5]], dtype=np.float64
        )  # cx, cy
        img_point = np.array(img_point, dtype=np.float64)
        img_point -= c_offset
        img_point = np.array(
            [img_point[0], img_point[1], cam_internal_param[0]], dtype=np.float64
        )
        img_point /= img_point[1]
        img_point *= cam_z_height
        return img_point

    def world_to_img_point(
        self, world_point: list, cam_internal_param: list, cam_z_height=1.0
    ):
        """Convert real world coordinate to image point

        Args:
            world_point (list): _description_
            cam_internal_param (list): _description_
            cam_z_height (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        x = (world_point[0] * cam_internal_param[0]) / (
            world_point[2] + 1e-10
        ) + cam_internal_param[2]
        y = (world_point[1] * cam_internal_param[4]) / (
            world_point[2] + 1e-10
        ) + cam_internal_param[5]
        return [x, y]

    def calc_width_point(self, coordinates, height):
        """?

        Args:
            coordinates (_type_): _description_
            height (_type_): _description_

        Returns:
            _type_: _description_
        """
        point_1, point_2 = coordinates[0], coordinates[1]
        if (point_1[0] - point_2[0]) == 0:
            return point_1[0]
        a = (point_1[1] - point_2[1]) / (point_1[0] - point_2[0])
        b = point_1[1] - a * point_1[0]
        point_x = (height - b) / a
        return point_x

    def single_point_crop_aug(
        self,
        image_path: str,
        coords: list,
        point_idx: int,
        is_debug: bool,
    ):
        image = Image.open(image_path)
        w, h = image.size
        window_size = [int(w * 0.8), int(h * 0.8)]
        slide_size = [int(w * 0.1), int(h * 0.1)]
        img_name = self.dataset_cropped_img_path + image_path.split("/")[-1]

        for cropping_idx in tqdm.tqdm(range(9), leave=False):
            v_slide_idx, h_slide_idx = divmod(cropping_idx, 3)
            save_name = img_name.replace(
                ".png",
                "_" + str(cropping_idx) + "_" + str(self.dataset_path_idx) + ".png",
            )

            left_width = slide_size[0] * h_slide_idx
            upper_height = slide_size[1] * v_slide_idx
            right_width = window_size[0] + left_width
            lower_height = window_size[1] + upper_height
            cropped_img = image.crop(
                (left_width, upper_height, right_width, lower_height)
            )

            shifted_coords = [coords[0] - left_width, coords[1] - upper_height]

            if not self._is_img_in_range(
                left_width, right_width, upper_height, lower_height, coords
            ):
                continue

            cropped_img.save(save_name)
            shifted_norm_coords = self._normalize_coordinate(
                window_size[0], window_size[1], shifted_coords
            )
            self.dataset_pair_list.append([save_name, shifted_norm_coords])

            if is_debug:
                self.draw_single_point(
                    save_name,
                    shifted_coords,
                    self.dataset_debug_img_path
                    + "debug_crop_"
                    + str(point_idx)
                    + "_"
                    + str(cropping_idx)
                    + "_"
                    + str(self.dataset_path_idx)
                    + ".png",
                )

    def multi_path_crop_aug(
        self,
        image_path: str,
        path_list: list,
        is_debug: bool,
        target_index: int,
        distance_threshold: float,
    ):
        """Cropping augmentation for multiple path

        Args:
            image_path (str): _description_
            path_list (list): _description_
            is_debug (bool): _description_
            target_index (int): _description_
            distance_threshold (float): _description_
        """
        image = Image.open(image_path)
        window_size = [int(image.width * 0.8), int(image.height * 0.8)]
        move_window = [int(image.width * 0.1), int(image.height * 0.1)]
        image_name = self.dataset_cropped_img_path + image_path.split("/")[-1]

        for loop_num in tqdm.tqdm(range(9), leave=False):
            cropped_img_save_name = image_name.replace(
                ".png",
                "_" + str(loop_num) + "_" + str(self.dataset_path_idx) + ".png",
            )
            vertical_num, horizon_num = divmod(loop_num, 3)
            crop_image = image.crop(
                (
                    move_window[0] * horizon_num,  # left
                    move_window[1] * vertical_num,  # upper
                    window_size[0] + move_window[0] * horizon_num,  # right
                    window_size[1] + move_window[1] * vertical_num,  # lower
                )
            )

            coords_list = self.gen_multiple_path_edge_points(
                path_list,
                move_window[0] * horizon_num,
                window_size[0] + move_window[0] * horizon_num,
                window_size[1] + move_window[1] * vertical_num,
                distance_threshold,
            )

            # if the output list is empty
            if not coords_list:
                continue

            shifted_coords_norm = []
            shifted_coords_list = []
            dataset_pair = [cropped_img_save_name]
            crop_image.save(cropped_img_save_name)
            for coords in coords_list:
                shifted_coords = [
                    [
                        coords[0][0] - move_window[0] * horizon_num,
                        window_size[1],
                    ],
                    [
                        coords[1][0] - move_window[0] * horizon_num,
                        coords[1][1] - move_window[1] * vertical_num,
                    ],
                ]

                shifted_coords_norm = [
                    [(shifted_coords[0][0]) / window_size[0], 1.0],
                    [
                        (shifted_coords[1][0]) / window_size[0],
                        shifted_coords[1][1] / window_size[1],
                    ],
                ]

                shifted_coords_list.append(shifted_coords)
                dataset_pair += shifted_coords_norm

            # padding with last coords
            for _ in range(0, self.MAX_BRANCH_NUM - len(coords_list)):
                dataset_pair += shifted_coords_norm

            self.dataset_pair_list.append(dataset_pair)
            if is_debug:
                self.draw_cropped_multi_line(
                    cropped_img_save_name,
                    shifted_coords_list,
                    target_index,
                    loop_num,
                    self.dataset_debug_img_path,
                )

    def real_world_based_crop_aug(
        self, image_path: str, path_list, save_flag, index, distance_threshold
    ):
        """_summary_
        Create 9 cropped image from a original one.

        Args:
            image_path (str): image_path from raw_dataset
            path_list (list): path_list (including img-path)
            save_flag (_type_): _description_
            path_name (_type_): _description_
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        image = Image.open(image_path)
        window_size = [int(image.width * 0.8), int(image.height * 0.8)]
        move_window = [int(image.width * 0.1), int(image.height * 0.1)]
        coordinate = []
        image_name = self.dataset_cropped_img_path + image_path.split("/")[-1]
        for roop_num in tqdm.tqdm(range(9), leave=False):
            vertical_num, horizon_num = divmod(roop_num, 3)
            crop_image = image.crop(
                (
                    move_window[0] * horizon_num,  # left
                    move_window[1] * vertical_num,  # upper
                    window_size[0] + move_window[0] * horizon_num,  # right
                    window_size[1] + move_window[1] * vertical_num,  # lower
                )
            )
            coordinate.append(
                self.gen_real_world_based_edge_points(
                    path_list,
                    move_window[0] * horizon_num,
                    window_size[0] + move_window[0] * horizon_num,
                    window_size[1] + move_window[1] * vertical_num,
                    distance_threshold,
                )
            )

            if len(coordinate[roop_num]) == 2:
                coordinate[roop_num] = [
                    [
                        coordinate[roop_num][0][0] - move_window[0] * horizon_num,
                        window_size[1],
                    ],
                    [
                        coordinate[roop_num][1][0] - move_window[0] * horizon_num,
                        coordinate[roop_num][1][1] - move_window[1] * vertical_num,
                    ],
                ]
                coords_norm = [
                    [(coordinate[roop_num][0][0]) / window_size[0], 1.0],
                    [
                        (coordinate[roop_num][1][0]) / window_size[0],
                        coordinate[roop_num][1][1] / window_size[1],
                    ],
                ]

                # save cropped image and append to dataset list
                crop_image.save(
                    image_name.replace(
                        ".png",
                        "_" + str(roop_num) + "_" + str(self.dataset_path_idx) + ".png",
                    )
                )
                self.dataset_pair_list.append(
                    [
                        image_name.replace(
                            ".png",
                            "_"
                            + str(roop_num)
                            + "_"
                            + str(self.dataset_path_idx)
                            + ".png",
                        ),
                        # coordinate[roop_num]
                        coords_norm[0],
                        coords_norm[1],
                    ]
                )
                if save_flag:
                    self.draw_crop_image(
                        image_name.replace(
                            ".png",
                            "_"
                            + str(roop_num)
                            + "_"
                            + str(self.dataset_path_idx)
                            + ".png",
                        ),
                        coordinate[roop_num],
                        index,
                        roop_num,
                        self.dataset_debug_img_path,
                    )
        # return image_path_list, coordinate_norm

    def crop_augmentation(self, image_path, path_list, save_flag, index):
        """_summary_
        Create 9 cropped image from a original one.

        Args:
            image_path (str): image_path from raw_dataset
            path_list (list): path_list (including img-path)
            save_flag (_type_): _description_
            path_name (_type_): _description_
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        image = Image.open(image_path)
        window_size = [int(image.width * 0.8), int(image.height * 0.8)]
        move_window = [int(image.width * 0.1), int(image.height * 0.1)]
        coordinate = []
        image_name = self.dataset_cropped_img_path + image_path.split("/")[-1]
        self.debug_point = []
        for roop_num in tqdm.tqdm(range(9), leave=False):
            vertical_num, horizon_num = divmod(roop_num, 3)
            crop_image = image.crop(
                (
                    move_window[0] * horizon_num,
                    move_window[1] * vertical_num,
                    window_size[0] + move_window[0] * horizon_num,
                    window_size[1] + move_window[1] * vertical_num,
                )
            )
            coordinate.append(
                self.gen_img_based_edge_points(
                    path_list,
                    move_window[0] * horizon_num,
                    window_size[0] + move_window[0] * horizon_num,
                    window_size[1] + move_window[1] * vertical_num,
                )
            )
            if len(coordinate[roop_num]) == 2:
                coordinate[roop_num] = [
                    [
                        coordinate[roop_num][0][0] - move_window[0] * horizon_num,
                        window_size[1],
                    ],
                    [
                        coordinate[roop_num][1][0] - move_window[0] * horizon_num,
                        int(window_size[1] * 0.8),
                    ],
                ]
                coords_norm = [
                    [(coordinate[roop_num][0][0]) / window_size[0], 1.0],
                    [(coordinate[roop_num][1][0]) / window_size[0], 0.8],
                ]

                # save cropped image and append to dataset list
                crop_image.save(
                    image_name.replace(
                        ".png",
                        "_" + str(roop_num) + "_" + str(self.dataset_path_idx) + ".png",
                    )
                )
                self.dataset_pair_list.append(
                    [
                        image_name.replace(
                            ".png",
                            "_"
                            + str(roop_num)
                            + "_"
                            + str(self.dataset_path_idx)
                            + ".png",
                        ),
                        # coordinate[roop_num]
                        coords_norm[0],
                        coords_norm[1],
                    ]
                )
                if save_flag:
                    self.draw_crop_image(
                        image_name.replace(
                            ".png",
                            "_"
                            + str(roop_num)
                            + "_"
                            + str(self.dataset_path_idx)
                            + ".png",
                        ),
                        coordinate[roop_num],
                        index,
                        roop_num,
                        self.dataset_debug_img_path,
                    )
            else:
                pass

    # def

    def separate_train_test(self):
        """_summary_
        separate the dataset into train and test by popping test_record randomly from self.dataset_pair_list
        """
        test_data_size = int(
            len(self.dataset_pair_list) * (1.0 - self.dataset_train_ratio)
        )

        self.test_dataset_size = test_data_size
        self.train_dataset_size = len(self.dataset_pair_list) - self.test_dataset_size
        for _ in range(test_data_size):
            target_idx = random.randint(0, len(self.dataset_pair_list) - 1)
            self.dataset_validation_pair_list.append(
                self.dataset_pair_list.pop(target_idx)
            )

    def write_out_debug_point(self, image, name, point_list):
        """Output debug path-point

        Args:
            image (_type_): _description_
            name (_type_): _description_
            point_list (_type_): _description_
        """
        canvas = ImageDraw.Draw(image)
        ds_range_start_point = [0, int(image.height * 0.8)]
        ds_range_end_point = [image.width, int(image.height * 0.8)]
        for point in point_list:
            canvas.ellipse(
                (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                fill=(255, 0, 0),
            )
        canvas.line(
            (
                ds_range_start_point[0],
                ds_range_start_point[1],
                ds_range_end_point[0],
                ds_range_end_point[1],
            ),
            fill=(0, 0, 255),
            width=3,
        )
        image.save(self.dataset_debug_point_img_path + name.split("/")[-1])

    def write_out_debug(self):
        """Output debug line."""
        for dataset_pair in self.dataset_pair_list:
            print("Processing img {}", dataset_pair[0])
            image = Image.open(dataset_pair[0])
            canvas = ImageDraw.Draw(image)
            canvas.ellipse(
                (
                    dataset_pair[1][0][0] - 3,
                    dataset_pair[1][0][1] - 3,
                    dataset_pair[1][0][0] + 3,
                    dataset_pair[1][0][1] + 3,
                ),
                fill=(255, 0, 0),
            )
            canvas.ellipse(
                (
                    dataset_pair[1][1][0] - 3,
                    dataset_pair[1][1][1] - 3,
                    dataset_pair[1][1][0] + 3,
                    dataset_pair[1][1][1] + 3,
                ),
                fill=(255, 0, 0),
            )
            canvas.line(
                (
                    dataset_pair[1][0][0],
                    dataset_pair[1][0][1],
                    dataset_pair[1][1][0],
                    dataset_pair[1][1][1],
                ),
                fill=(255, 0, 0),
                width=4,
            )
            image.save(self.dataset_debug_img_path + dataset_pair[0].split("/")[-1])

    def write_out_dataset(self):
        """write out train/validation dataset"""
        header = ["image_path"]
        if self.is_multi_path_labeling:
            for idx in range(0, self.MAX_BRANCH_NUM):
                if not self.is_only_point_dataset:
                    header.append("start_coordinate_" + str(idx))
                header.append("end_coordinate_" + str(idx))
        else:
            if not self.is_only_point_dataset:
                header = ["image_path", "start_coordinate", "end_coordinate"]
            else:
                header = ["image_path", "end_coordinate"]

        with open(self.dataset_train_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for dataset_pair in self.dataset_pair_list:
                writer.writerow(dataset_pair)

        with open(self.dataset_validation_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for dataset_pair in self.dataset_validation_pair_list:
                writer.writerow(dataset_pair)

    def save_image_data(self, save_dir):
        """save image

        Args:
            save_dir (_type_): _description_
        """
        for index, image_path in enumerate(self.image_path_list):
            data = Image.open(image_path)
            data.save(
                self.parent
                + save_dir
                + str(index)
                + "_"
                + str(self.dataset_path_idx)
                + ".png"
            )

    def draw_image(self, index, coordinate, save_path):
        """draw line

        Args:
            index (_type_): _description_
            coordinate (_type_): _description_
            save_path (_type_): _description_
        """
        image = Image.open(self.image_path_list[index])
        canvas = ImageDraw.Draw(image)
        canvas.ellipse(
            (
                coordinate[0][0] - 3,
                coordinate[0][1] - 3,
                coordinate[0][0] + 3,
                coordinate[0][1] + 3,
            ),
            fill=(255, 0, 0),
        )
        canvas.ellipse(
            (
                coordinate[1][0] - 3,
                coordinate[1][1] - 3,
                coordinate[1][0] + 3,
                coordinate[1][1] + 3,
            ),
            fill=(255, 0, 0),
        )
        canvas.line(
            (coordinate[0][0], coordinate[0][1], coordinate[1][0], coordinate[1][1]),
            fill=(255, 0, 0),
            width=4,
        )
        image.save(
            save_path
            + "debug_"
            + str(index)
            + "_"
            + str(self.dataset_path_idx)
            + ".png"
        )

    def draw_multi_line(self, index, coordinate_list: list, save_path):
        """draw multiple path line

        Args:
            index (_type_): _description_
            coordinate_list (list): _description_
            save_path (_type_): _description_
        """
        image = Image.open(self.image_path_list[index])
        canvas = ImageDraw.Draw(image)
        for coords in coordinate_list:
            canvas.ellipse(
                (
                    coords[0][0] - 3,
                    coords[0][1] - 3,
                    coords[0][0] + 3,
                    coords[0][1] + 3,
                ),
                fill=(255, 0, 0),
            )
            canvas.ellipse(
                (
                    coords[1][0] - 3,
                    coords[1][1] - 3,
                    coords[1][0] + 3,
                    coords[1][1] + 3,
                ),
                fill=(255, 0, 0),
            )
            canvas.line(
                (coords[0][0], coords[0][1], coords[1][0], coords[1][1]),
                fill=(255, 0, 0),
                width=4,
            )

        image.save(
            save_path
            + "debug_"
            + str(index)
            + "_"
            + str(self.dataset_path_idx)
            + ".png"
        )

    def draw_single_point(
        self,
        img_path: str,
        coords: list,
        save_path: str,
    ):
        img = Image.open(img_path)
        canvas = ImageDraw.Draw(img)

        canvas.ellipse(
            (coords[0] - 5, coords[1] - 5, coords[0] + 5, coords[1] + 5),
            fill=(255, 0, 0),
        )

        img.save(save_path)

    def draw_cropped_multi_line(
        self,
        image_path: str,
        coordinate_list: list,
        index: int,
        loop: int,
        save_path: str,
    ):
        """draw cropped multiple path line

        Args:
            image_path (str): _description_
            coordinate_list (list): _description_
            index (int): _description_
            loop (int): _description_
            save_path (str): _description_
        """
        image = Image.open(image_path)
        canvas = ImageDraw.Draw(image)

        for coords in coordinate_list:
            canvas.ellipse(
                (
                    coords[0][0] - 3,
                    coords[0][1] - 3,
                    coords[0][0] + 3,
                    coords[0][1] + 3,
                ),
                fill=(255, 0, 0),
            )
            canvas.ellipse(
                (
                    coords[1][0] - 3,
                    coords[1][1] - 3,
                    coords[1][0] + 3,
                    coords[1][1] + 3,
                ),
                fill=(255, 0, 0),
            )
            canvas.line(
                (coords[0][0], coords[0][1], coords[1][0], coords[1][1]),
                fill=(255, 0, 0),
                width=4,
            )
        image.save(
            save_path
            + "debug_crop_"
            + str(index)
            + "_"
            + str(loop)
            + "_"
            + str(self.dataset_path_idx)
            + ".png"
        )

    def draw_crop_image(self, image_path, coordinate, index, roop, save_path):
        """Draw cropped path line

        Args:
            image_path (_type_): _description_
            coordinate (_type_): _description_
            index (_type_): _description_
            roop (_type_): _description_
            save_path (_type_): _description_
        """
        image = Image.open(image_path)
        canvas = ImageDraw.Draw(image)
        canvas.ellipse(
            (
                coordinate[0][0] - 3,
                coordinate[0][1] - 3,
                coordinate[0][0] + 3,
                coordinate[0][1] + 3,
            ),
            fill=(255, 0, 0),
        )
        canvas.ellipse(
            (
                coordinate[1][0] - 3,
                coordinate[1][1] - 3,
                coordinate[1][0] + 3,
                coordinate[1][1] + 3,
            ),
            fill=(255, 0, 0),
        )
        canvas.line(
            (coordinate[0][0], coordinate[0][1], coordinate[1][0], coordinate[1][1]),
            fill=(255, 0, 0),
            width=4,
        )
        image.save(
            save_path
            + "debug_crop_"
            + str(index)
            + "_"
            + str(roop)
            + "_"
            + str(self.dataset_path_idx)
            + ".png"
        )


def arg_manage():
    parser = argparse.ArgumentParser(description="Create corresponding datasets")
    parser.add_argument("path", type=str, help="Describe the target CSV file")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-s", "--save_name")
    parser.add_argument("-p", "--parent", default="")
    argument = parser.parse_args()
    return argument


if __name__ == "__main__":
    # arg = arg_manage()
    # dataset_Generator = DatasetGenerator(["/raw_dataset/sakaki_tr1/sakaki_tr1.csv","/raw_dataset/sakaki_tr2/sakaki_tr2.csv"],
    #                               "fix_sakaki_dataset_20230920_ratio_7_len_2",
    #                               dataset_train_ratio=0.7,
    #                               is_img_based_labeling=False,
    #                               dataset_distance_threshold=2.0)
    # dataset_generator = DatasetGenerator(
    #     raw_dataset_dir_list = ["/raw_dataset/sakaki_20230705",],
    #     dataset_save_name="sakaki_20230705_r7_l2",
    #     dataset_train_ratio=0.7,
    #     is_img_based_labeling=False,
    #     dataset_distance_threshold=2.0
    #     )

    # dataset_generator = DatasetGenerator(
    #     raw_dataset_dir_list=[
    #         "/raw_dataset/sakaki_tr1_20221221",
    #         "/raw_dataset/sakaki_tr2_20221221",
    #         # "/raw_dataset/sakaki_tr4_curv",
    #         # "/raw_dataset/sakaki_tr5_curv",
    #     ],
    #     dataset_save_base_name="sakaki_ds_l2_",
    #     dataset_train_ratio=0.7,
    #     is_img_based_labeling=False,
    #     dataset_distance_threshold=2.0,
    # )
    # dataset_Generator = DatasetGenerator(
    #     ["/raw_dataset/sakaki_tr1/sakaki_tr1.csv"],
    #     # ["/raw_dataset/sakaki_test_0912.csv"],
    #     "len_2.0_test",
    # )

    # dataset_generator.generate()

    dataset_generator = DatasetGenerator(
        raw_dataset_dir_list=[
            "/raw_dataset/sakaki_20231112_jct1_0",
            "/raw_dataset/sakaki_20231112_jct1_1",
            "/raw_dataset/sakaki_20231112_jct1_2",
            "/raw_dataset/sakaki_jct2_0_20231103",
            "/raw_dataset/sakaki_jct2_1_20231103",
            "/raw_dataset/sakaki_jct2_2_20231103",
            "/raw_dataset/sakaki_jct3_0_20231103",
            "/raw_dataset/sakaki_jct3_1_20231103",
            "/raw_dataset/sakaki_jct3_2_20231103",
        ],
        dataset_save_base_name="sakaki_jct_point_ds",
        dataset_train_ratio=0.9,
        is_only_point_dataset=True,
        is_const_point_labeling=True,
        # is_img_based_labeling=False,
        # dataset_distance_threshold=1.5,
        # is_multi_path_labeling=True,
        # max_branch_num=2,
    )

    # dataset_generator.test_generate()
    dataset_generator.generate()
