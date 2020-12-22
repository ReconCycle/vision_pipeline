from config import *
import numpy as np
import cv2


if __name__ == '__main__':
    img = cv2.imread('data/calibration_undistorted_19-11-2020/0.png')
    cv2.imshow('img',img)
    waitkey = cv2.waitKey(3000)