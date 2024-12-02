from model.predict_model import PredictModel

import numpy as np
import cv2

def normalize_img(img: cv2.Mat):
    ret_img = cv2.resize(img, (240, 128))
    ret_img = np.array(ret_img, np.float32)
    ret_img /= 255.0
    return ret_img

if __name__=="__main__":
    model = PredictModel(model_name="CCP",
                         weight_pass="",
                         estimation=True)
    model = model.get_model()
    file_path = ""
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = normalize_img(img)
    
    img_norm = img_norm.reshape(1, 128, 240, 3)
    
    
    