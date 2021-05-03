# from dlc.config import *
import os
from dlc.infer import Inference
from image_calibration import ImageCalibration
from work_surface_detection import WorkSurfaceDetection
import numpy as np

if __name__ == '__main__':
    full_path = os.getcwd()
    config_path_kalo = os.path.join(full_path, 'data_full/dlc/kalo_v2-sebastian-2020-11-04/config.yaml')
    config_path_work_surface = os.path.join(full_path, 'data_full/dlc/work_surface-sebastian-2020-11-19/config.yaml')

    calibration = ImageCalibration(calibration_file="data/camera_calibration/calibration_1450x1450.yaml")

    camera_matrix = np.array(calibration.calibration['camera_matrix'])
    dist_coefs = np.array(calibration.calibration['dist_coefs'])

    coords = WorkSurfaceDetection("/home/sruiz/datasets/deeplabcut/kalo_v2_imgs_20-11-2020/0.png")

    inference = Inference(config_path_kalo)

    df_x, df_y, df_likelihood, body_parts, bpts2connect = inference.infer_from_img("/home/sruiz/datasets/deeplabcut/kalo_v2_imgs_20-11-2020", coords,
                                                                                   calibration=(camera_matrix, dist_coefs))
    
    # print("df_x", df_x)
    # print("df_y", df_y)
    # print("df_likelihood", df_likelihood)
    # print("body_parts", body_parts)
