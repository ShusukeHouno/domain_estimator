#!/usr/bin/env python

from .cnn.path_estimation import (
    PathEstimationCCP,
    PathEstimationCP,
    PathEstimationCCP_MLT2,
    PathEstimationCCP_MLT2_TANH,
    PathEstimationCCCP_MLT2,
    PathEstimationCCP_JCT
)


class PredictModel:
    def __init__(self, model_name, weight_path="", estimation=False):
        self.model_name = model_name
        self.weight_path = weight_path
        self.model = self.init_model(model_name, weight_path, estimation)

    def init_model(self, model_name, weight_path, estimation):
        if model_name == "CCP":
            return PathEstimationCCP(weight_path, estimation)
        #   model = PathEstimationCCP(weight_path, estimation)
        #   self.model = model.set_model_layer() # 重み引き継いでないやんけボケ
        elif model_name == "CP":
            return PathEstimationCP(weight_path)

        elif model_name == "CCP_MLT2":
            return PathEstimationCCP_MLT2(weight_path, estimation)
        elif model_name == "CCP_MLT2_TANH":
            return PathEstimationCCP_MLT2_TANH(weight_path, estimation)
        elif model_name == "CCCP_MLT2":
            return PathEstimationCCCP_MLT2(weight_path, estimation)
        elif model_name == "CCP_JCT":
            return PathEstimationCCP_JCT(weight_path, estimation)
        else:
            print("[Error] choose model name : CCP or CP")

    def get_model(self):
        return self.model.get_model()

    def get_weight_name(self):
        return self.weight_path

    def save_weight(self, parent_path, epoch, batch):
        self.model.save_model(
            parent_path
            + "model_name_"
            + model_name
            + "_epochs"
            + str(epoch)
            + "_batch"
            + str(batch)
        )
