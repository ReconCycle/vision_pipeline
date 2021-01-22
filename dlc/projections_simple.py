from config import *
import numpy as np
import cv2
from calibration import Calibration


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    print("corner, tuple(imgpts[0].ravel())", corner, tuple(imgpts[0].ravel()))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


if __name__ == '__main__':
    calibration = Calibration(calibration_file="data/kalo_v2_calibration/calibration_1450x1450.yaml",
                              basler_config_file="basler_config.yaml")


    mtx = np.array(calibration.calibration['camera_matrix'])
    dist = np.array(calibration.calibration['dist_coefs'])
    print("mtx", mtx)
    print("dist", dist)

    pattern_size = (8, 6)


    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    for fname in get_images('data/calibration_undistorted_19-11-2020'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=cv2.CALIB_CB_FILTER_QUADS)

        print("corners", corners)
        print("ret:", ret)

        print("fname", fname)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            print("corners2", corners2)

            objp = np.zeros((6*8,3), np.float32)
            objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            print("rvecs", rvecs)
            print("tvecs", tvecs)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            img = draw(img,corners2,imgpts)
            cv2.imshow('img',img)
            k = cv2.waitKey(0) & 0xff
            if k == 's':
                cv2.imwrite(fname[:6]+'asdf.png', img)

    cv2.destroyAllWindows()