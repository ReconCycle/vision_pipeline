from config import *
import numpy as np
import cv2
from calibration import Calibration


if __name__ == '__main__':
    calibration = Calibration(calibration_file="data/kalo_v2_calibration/calibration_1450x1450_undistorted.yaml",
                              basler_config_file="basler_config.yaml")


    camera_matrix = np.array(calibration.calibration['camera_matrix'])
    dist_coefs = np.array(calibration.calibration['dist_coefs'])
    print("camera_matrix", camera_matrix)
    print("dist_coefs", dist_coefs)

    pattern_size = (8, 6)

    # 3D model points in meters, left-handed cartesian coords (x, y, z)
    model_points = np.array([
        (0.0, 0.0, 0.025),  # corner1
        (0.04, 0.0, 0.025),  # corner2
        (0.0, 0.115, 0.025),  # corner3
        (0.04, 0.115, 0.025),  # corner4
        (0.04, 0.0, 0.0),  # corner5
        (0.0, 0.0, 0.0),  # corner6
        (0.04, 0.115, 0.0),  # corner7
        (0.0, 0.115, 0.0),  # corner8
        (0.02, 0.025, 0.025),  # screen_middle
        (0.02, 0.04, 0.025),  # sensor_middle
        (0.02, 0.04, 0.005),  # battery_middle
        (0.02, 0.05525, 0.005),  # object_centre
    ])

    df_x = np.array([[564.15246582],
                    [508.24884033],
                    [389.08557129],
                    [333.35458374],
                    [565.06524658],
                    [563.80462646],
                    [243.47848511],
                    [244.21890259],
                    [490.43481445],
                    [470.85235596],
                    [507.33074951]])

    df_y = np.array([[421.24169922],
                     [491.97494507],
                     [240.64611816],
                     [295.54760742],
                     [437.50836182],
                     [434.92254639],
                     [379.83972168],
                     [387.31625366],
                     [409.68798828],
                     [390.30456543],
                     [420.28765869]])

    df_likelihood = np.array([[0.25643784],
                            [0.01832396],
                            [1.],
                            [1.],
                            [0.54634094],
                            [0.99968731],
                            [0.23692116],
                            [0.45416582],
                            [0.99998903],
                            [0.99999928],
                            [0.04214257]])

    image_points = np.hstack((df_x, df_y))
    print("image_points", image_points)

    body_parts = np.array(['corner1', 'corner2', 'corner3', 'corner4', 'corner5', 'corner6', 'corner7', 'corner8', 'screen_middle', 'sensor_middle', 'battery_middle'])

    # for bpindex in range(len(body_parts)):
    #     if df_likelihood[bpindex, 0] > 0.95:
    #         model_

    print("df_likelihood.shape", df_likelihood[:, 0].shape)
    mask = np.nonzero(df_likelihood[:, 0] > 0.95)
    print("mask", mask, mask[0], len(mask[0]))

    image_points_masked = image_points[mask]
    model_points_masked = model_points[mask]
    body_parts_masked = body_parts[mask]
    print("image_points_masked", image_points_masked)
    print("body_parts_masked", body_parts_masked)

    axis = np.float32([[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]]) + model_points[-1]
    print("model_points[-1]", model_points[-1])
    print("axis", axis.shape, axis)

    for fname in ['data/video_20-11-2020/0.png']:
        img = cv2.imread(fname)

        print("fname", fname)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(model_points_masked, image_points_masked, camera_matrix, dist_coefs)
        # _, rvecs, tvecs, inliers = cv2.solvePnP(model_points_masked, image_points_masked, camera_matrix, dist_coefs, flags=cv2.SOLVEPNP_ITERATIVE)

        print("rvecs", rvecs)
        print("tvecs", tvecs)

        # project 3D points to image plane
        projected_axis, _ = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coefs)
        projected_model_points, _ = cv2.projectPoints(model_points, rvecs, tvecs, camera_matrix, dist_coefs)

        print("projected_axis", projected_axis)
        print("projected_model_points", projected_model_points.shape, projected_model_points)

        for p in image_points_masked:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        for p in projected_model_points[:, 0]:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

        def draw(img, centre_point, axis_points):
            the_centre_point = tuple(centre_point.ravel().astype(int))
            print("centre_point", the_centre_point, tuple(axis_points[0].ravel().astype(int)))
            print("axis_points", axis_points)
            img = cv2.line(img, the_centre_point, tuple(axis_points[0].ravel().astype(int)), (255, 0, 0), 5)
            img = cv2.line(img, the_centre_point, tuple(axis_points[1].ravel().astype(int)), (0, 255, 0), 5)
            img = cv2.line(img, the_centre_point, tuple(axis_points[2].ravel().astype(int)), (0, 0, 255), 5)
            return img


        img = draw(img, projected_model_points[-1], projected_axis)
        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'asdf.png', img)

    cv2.destroyAllWindows()