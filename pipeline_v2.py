import os
import numpy as np
import time
import cv2
from rich import print

from image_calibration import ImageCalibration
from work_surface_detection_opencv import WorkSurfaceDetection
from object_detection import ObjectDetection

from helpers import scale_img, get_images
from config import load_config

def quaternion_multiply(quaternion1, quaternion0):
        x0, y0, z0, w0 = quaternion0
        x1, y1, z1, w1 = quaternion1

        # This quat is W X Y Z
        out_quat = np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                     x1*w0 + y1*z0 - z1*y0 + w1*x0,
                    -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                     x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)

        # This quat is X Y Z W
        out = out_quat[[1,2,3,0]]
        return out

class Pipeline:
    def __init__(self):
        self.config = load_config()
        print("config", self.config)
        
        # 1. load camera calibration files
        self.calibration = ImageCalibration(self.config.camera)

        # 2. work surface coordinates, will be initialised on first received image
        self.worksurface_detection = None

        # 3. object detection
        self.object_detection = ObjectDetection(self.config.obj_detection)


    def process_img(self, img, fps=None):
        if isinstance(img, str):
            print("\nimg path:", img)
            img = cv2.imread(img)
        
        if self.worksurface_detection is None:
            print("detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img)
            # self.worksurface_detection = WorkSurfaceDetection(img, self.config.dlc)
        
        labelled_img, detections = self.object_detection.get_prediction(img, self.worksurface_detection, fps)

        for detection in detections:
            corners = np.array(detection.obb_corners)
            distances = []
            # Logger.loginfo("{}".format(corners))
            first_corner = corners[0]

            for ic, corner in enumerate(corners):
                distances.append(np.linalg.norm(corner - first_corner))
            # Logger.loginfo("Distances: {}".format(distances))
            distances = np.array(distances)
            idx_edge = distances.argsort()[-2]
            # Logger.loginfo("Index of edge: {}".format(idx_edge))
            second_corner = corners[idx_edge]

            highest_y = np.argmax([first_corner[1], second_corner[1]])
            if highest_y == 0:
                vector_1 = first_corner - second_corner
            elif highest_y == 1:
                vector_1 = second_corner - first_corner

            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = np.array([0, 1])
            # Logger.loginfo("Vectors: {}, {}".format(vector_1, unit_vector_2))

            angle = (np.arctan2(unit_vector_1[1], unit_vector_1[0]) -
                        np.arctan2(unit_vector_2[1], unit_vector_2[0]))


            # If angle is too negative, add 180 degrees
            if (angle * 180 / np.pi) < -30:
                angle = angle + np.pi
            # Logger.loginfo("Angle: {}".format(angle * 180 / np.pi))

            # Below code works but z-axis is incorrect, should be rotated by 180 degs
            angle = -angle

            rot_quat = np.concatenate((np.sin(angle/2)*np.array([0,0,1]),
                                        np.array([np.cos(angle/2)])))

            #rot_quat = np.concatenate((np.sin(angle/2)*np.array([0,0,-1]),
            #                           np.array([-np.cos(angle/2)])))

            #Rotate around x-axis by 180 degs
            rot_quat = quaternion_multiply(np.array([0,1,0,0]), rot_quat)

            #Rotate around z-axis by 180 degs
            #rot_quat = quaternion_multiply(np.array([0,0,1,0]), rot_quat)

            detection.obb_rot_quat = rot_quat.tolist()
            # detection.obb_rot_quat = np.array([[1,0,0,0]]).tolist()
        
        return labelled_img, detections
    

if __name__ == '__main__':

    pipeline = Pipeline()

    # pipeline
    show_imgs = False

    # Iterate over images and run:
    # we can use a directory here or a single image /163.png
    # img_path = "data_full/dlc/dlc_work_surface_jsi_05-07-2021/labeled-data/raw_work_surface_jsi_08-07-2021"
    # img_path = "/home/sruiz/datasets/reconcycle/2022-02-17_kalo_tracking_2"
    # img_path = "/home/sruiz/datasets/reconcycle/2022-02-17_kalo_tracking/"
    img_path = "/home/sruiz/datasets/reconcycle/2022-04-04_qundis_disassembly/"
    save_path = "./save_images" # set to None to not save
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs = get_images(img_path)

    for img_p in imgs:
        labeled_img, detections = pipeline.process_img(img_p)
        
        t_start = time.time()
        if show_imgs:
            cv2.imshow('labeled_img', scale_img(labeled_img))

        waitkey = cv2.waitKey(1)

        # break
        if waitkey == 27:
            break  # esc to quit
        elif waitkey == ord('p'): # pause
            cv2.waitKey(-1)  # wait until any key is pressed

        if save_path is not None:
            save_file_path = os.path.join(save_path, os.path.basename(img_p))
            print("saving!", save_file_path)
            cv2.imwrite(save_file_path, labeled_img)
            
        fps_imshow = 1.0 / (time.time() - t_start)
        print("fps_imshow", str(np.int(round(fps_imshow, 0))))
