import sys, os
# having trouble importing the yolact directory. Doing this as a workaround:
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deeplabcut'))
os.environ["DLClight"] = "True" # no gui
os.environ['KMP_WARNINGS'] = 'off' # some info about KMP_AFFINITY
from config import *
from infer import Inference
from image_calibration import ImageCalibration
from work_surface_detection import WorkSurfaceDetection
import numpy as np
from object_detection import ObjectDetection
from helpers import *
import graphics
import cv2
import torch
import time

if __name__ == '__main__':

    # pipeline
    show_imgs = False

    # 1. load calibration files
    # * this part works!
    # calibration = ImageCalibration(calibration_file="/home/sruiz/datasets/deeplabcut/kalo_v2_calibration/calibration_1450x1450_undistorted.yaml",
                                    # basler_config_file="config/basler_config.yaml")

    # camera_matrix = np.array(calibration.calibration['camera_matrix'])
    # dist_coefs = np.array(calibration.calibration['dist_coefs'])

    # 2. get work surface coordinates
    # todo: broken atm, missing the work-surface data
    # full_path = os.path.dirname(os.path.abspath(__file__))
    # config_path_work_surface = os.path.join(full_path, 'deeplabcut/work_surface-sebastian-2020-11-19/config.yaml')

    # coords = WorkSurfaceDetection(config_path_work_surface, "/home/sruiz/datasets/deeplabcut/data/video_20-11-2020/0.png")
    # 3. object detection

    object_detection = ObjectDetection() # *working

    # Iterate over images and run:
    img_path = "/home/sruiz/datasets/deeplabcut/kalo_v2_imgs_20-11-2020/163.png"
    save_path = '' # set to None to not save
    imgs = get_images(img_path)
    if show_imgs:
        cv2.namedWindow("labeled_img", cv2.WINDOW_NORMAL)
    frame_count = 0
    for img_p in imgs:
        with torch.no_grad():
            frame = torch.from_numpy(cv2.imread(img_p)).cuda().float()
        print("img_p", img_p)
        print("frame.shape", frame.shape)
        pred = object_detection.get_prediction(frame)
        labeled_img = graphics.get_labeled_img(pred, frame, None, None, undo_transform=False)
        print("labeled_img.shape", labeled_img.shape)

        if show_imgs:
            cv2.imshow('labeled_img', labeled_img)

        waitkey = cv2.waitKey(1)

        # break
        if waitkey == 27:
            break  # esc to quit
        elif waitkey == ord('p'): # pause
            cv2.waitKey(-1)  # wait until any key is pressed

        t_now = time.time()
        # only start calculating avg fps after the first frame
        if frame_count == 0:
            t_start = time.time()
        else:
            print("avg. FPS:", frame_count / (t_now - t_start))
        frame_count += 1

        if save_path is not None:
            save_file_path = os.path.join(save_path, os.path.basename(img_p))
            print("saving!", save_file_path)
            cv2.imwrite(save_file_path, labeled_img)



    # * old pipeline for deeplabcut detection
    # inference = Inference(config_path_kalo)
    # inference.infer_from_img("data/video_20-11-2020", coords)
    # df_x, df_y, df_likelihood, body_parts, bpts2connect = inference.infer_from_img("~/datasets/deeplabcut/video_20-11-2020", coords,
    #                                                                               calibration=(camera_matrix, dist_coefs))

    # test prints:
    # print("df_x", df_x)
    # print("df_y", df_y)
    # print("df_likelihood", df_likelihood)
    # print("body_parts", body_parts)
    # * end old code
    

