#!/usr/bin/bin python

import os
import csv
import tqdm
import numpy as np
import datetime
import random

# dataset is used to be CNN training
# ds -> test, train
# datasets are expected to be created from same raw_datasets
# only the csv data are saved
"""
ds_dict = {
    'ds_name0' : [
        [path, point0, point1, ...],
        ...
    ],
    'ds_name1' : [
        [path, point0, point1, ...],
        ...
    ],
    'ds_name2' : [
        [path, point0, point1, ...],
        ...
    ]
}
"""

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")


class Adjuster:
    GENERATED_TRAIN_DS = "generated_train_dataset.csv"
    GENERATED_TEST_DS = "generated_test_dataset.csv"

    def __init__(self, dataset_dir_list: list, dataset_train_ratio: int):
        self.dataset_list = dataset_dir_list
        self.loaded_ds: dict = self.load_dataset()

        self.adjust_base_ds: tuple = min(
            self.loaded_ds.items(), key=lambda ds: len(ds[1])
        )

        self.result_save_path = (
            "/tf/adjusted_dataset/"
            + datetime.datetime.now(JST).strftime("%Y%m%d%H%M")
            + "/"
        )

        os.umask(0)
        for ds_name in self.loaded_ds.keys():
            os.makedirs(self.result_save_path + ds_name, mode=0o777, exist_ok=True)

        self.dataset_train_ratio = dataset_train_ratio
        del self.loaded_ds[self.adjust_base_ds[0]]

    def adjust(self):
        # init dict
        adjust_base_ds_name = self.adjust_base_ds[0]
        adjusted_ds_dict: dict = {}
        adjusted_ds_dict[adjust_base_ds_name] = []

        for other_ds_name in self.loaded_ds.keys():
            adjusted_ds_dict[other_ds_name] = []

        process_bar = tqdm.tqdm(total=len(self.adjust_base_ds[1]))
        process_bar.set_description("Adjusting Process")
        for base_ds_row in self.adjust_base_ds[1]:
            base_img_path = base_ds_row[0].split("/")[-2] + "/" + base_ds_row[0].split("/")[-1]
            # print(base_img_path)
            delete_base_elem = False
            for other_ds in self.loaded_ds.items():
                # other_ds <- ('other_ds_name' : [[path, points,...], [path, points,...], ...])
                match_elem = list(
                    filter(lambda elem: (elem[0].split("/")[-2] + "/" + elem[0].split("/")[-1]) == base_img_path, other_ds[1])
                )
                if len(match_elem) > 0:
                    adjusted_ds_dict[other_ds[0]].append(match_elem[0])
                else:
                    delete_base_elem = True
            if not delete_base_elem:
                adjusted_ds_dict[adjust_base_ds_name].append(base_ds_row)
            process_bar.update(1)
        process_bar.close()

        for adjusted_ds in adjusted_ds_dict.items():
            result = tuple(
                [adjusted_ds[0], self.separate_train_validation(adjusted_ds[1])]
            )
            self.write_out_dataset(result)

    def separate_train_validation(self, target_ds: list) -> list:
        ds = target_ds.copy()
        validation_ds = []
        validation_data_size = int(len(ds) * (1.0 - self.dataset_train_ratio))
        train_data_size = len(ds) - validation_data_size
        for _ in range(validation_data_size):
            target_idx = random.randint(0, len(ds) - 1)
            validation_ds.append(ds.pop(target_idx))
        return [ds, validation_ds]

    def write_out_dataset(self, adjusted_ds: tuple):
        train_path = (
            self.result_save_path + adjusted_ds[0] + "/generated_train_dataset.csv"
        )
        validation_path = (
            self.result_save_path + adjusted_ds[0] + "/generated_validation_dataset.csv"
        )
        with open(train_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "start_coordinate", "end_coordinate"])
            for train_ds_pair_data in adjusted_ds[1][0]:
                writer.writerow(train_ds_pair_data)

        with open(validation_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "start_coordinate", "end_coordinate"])
            for validation_ds_pair_data in adjusted_ds[1][1]:
                writer.writerow(validation_ds_pair_data)

    def load_dataset(self) -> dict:
        all_loaded_ds = {}
        for ds in self.dataset_list:
            train_ds = ds + "/" + Adjuster.GENERATED_TRAIN_DS
            test_ds = ds + "/" + Adjuster.GENERATED_TEST_DS
            loaded_train_ds = self.load_csv(train_ds)
            loaded_test_ds = self.load_csv(test_ds)
            all_loaded_ds[ds] = loaded_train_ds + loaded_test_ds
        return all_loaded_ds

    def load_csv(self, ds) -> list:
        loaded_ds = []
        with open(ds, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for r_idx, row in enumerate(reader):
                row_data = []
                if r_idx != 0:
                    for idx, data in enumerate(row):
                        if data == "":
                            continue
                        if idx != 0:
                            # for decoding path-point data
                            data = eval(data)
                        row_data.append(data)
                    loaded_ds.append(row_data)
        return loaded_ds


if __name__ == "__main__":
    dataset_adjuster = Adjuster(
        dataset_dir_list=[
            "/generated_dataset/sakaki_ds_img_based_202310160038",
            "/generated_dataset/sakaki_ds_l2_202310160055",
        ],
        dataset_train_ratio=0.7,
    )

    dataset_adjuster.adjust()
