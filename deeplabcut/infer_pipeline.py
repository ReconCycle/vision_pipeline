from config import *
from infer import Inference
from calibration import Calibration
from coordinates import WorkSurfaceCoordinates
import numpy as np

if __name__ == '__main__':
    calibration = Calibration(calibration_file="data/kalo_v2_calibration/calibration_1450x1450_undistorted.yaml",
                              basler_config_file="basler_config.yaml")

    camera_matrix = np.array(calibration.calibration['camera_matrix'])
    dist_coefs = np.array(calibration.calibration['dist_coefs'])

    coords = WorkSurfaceCoordinates(config_path_work_surface, "data/video_20-11-2020/0.png")

    inference = Inference(config_path_kalo)
    # inference.infer_from_img("data/video_20-11-2020", coords)
    df_x, df_y, df_likelihood, body_parts, bpts2connect = inference.infer_from_img("data/video_20-11-2020", coords,
                                                                                   calibration=(camera_matrix, dist_coefs))
    # print("df_x", df_x)
    # print("df_y", df_y)
    # print("df_likelihood", df_likelihood)
    # print("body_parts", body_parts)
