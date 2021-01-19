from pypylon import pylon
import cv2
import os
import time
from calibration import *
import sys

def record_from_webcam(save_path, mirror=False):
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    files_in_dir = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    print("files_in_dir", files_in_dir)
    files_by_name_only = [i.split('.', 1)[0] for i in files_in_dir]
    filenames_as_ints = list(map(int, files_by_name_only))

    count = 0
    if len(filenames_as_ints) > 0:
        count = max(filenames_as_ints) + 1
    print("count = ", count)

    while True:
        ret_val, img = cap.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        waitkey = cv2.waitKey(1)
        if waitkey == 27:
            break  # esc to quit
        elif waitkey == 115:  # "s" key
            print("saving image", str(count))
            cv2.imwrite(os.path.join(save_path, str(count) + '.png'), img)
            count += 1

    cv2.destroyAllWindows()


def record_from_basler(save_path, record_video=False, undistort=True, save_all_imgs=False, limit_fps=None):
    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    files_in_dir = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    print("files_in_dir", files_in_dir)
    files_by_name_only = [i.split('.', 1)[0] for i in files_in_dir]
    filenames_as_ints = list(map(int, files_by_name_only))

    count = 0
    if len(filenames_as_ints) > 0:
        count = max(filenames_as_ints) + 1
    print("count = ", count)

    t0 = time.time()
    tstart = t0
    last_image_taken = 0

    if record_video:
        fps, duration = 5, 100
        video_resolution = (1450, 1450)
        # Define the codec and create VideoWriter object to save the video
        # fourcc = cv2.VideoWriter_fourcc(*'XVID') # very big compression ratio
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(os.path.join(save_path, 'output.avi'), fourcc, fps, video_resolution)  # native res: 3500, 2900
    else:
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()
            if undistort:
                img = calibration.undistort_imgs(img)  # resizes and undistorts image

            if record_video:
                t1 = time.time()
                diff = t1 - t0
                # if diff > 1/fps: # if you want to restrict fps.
                # print("img.shape", img.shape)
                print("time taken:", diff)
                # print("saving image", str(count))
                print(count/fps, "of", duration, "seconds")

                print("img.shape", img.shape)

                if img.shape[:2] != video_resolution:
                    print("img resolution not equal to video resolution!", img.shape[:2], video_resolution)

                if count < fps * duration:
                    video_writer.write(img)
                else:
                    print("STOP RECORDING>>>>")
                    break

                count += 1
                t0 = t1
            else:
                cv2.imshow('title', img)

            waitkey = cv2.waitKey(1)
            if waitkey == 27:
                break
            elif (waitkey == 115 and not record_video) or save_all_imgs:  # "s" key
                time_now = time.time()
                if (limit_fps and time_now > last_image_taken + 1/limit_fps) or limit_fps is None:
                    img_3 = np.zeros([512, 512, 3], dtype=np.uint8)
                    img_3.fill(255)
                    # or img[:] = 255
                    cv2.imshow('title', img_3)

                    sys.stdout.write('\a')
                    print("saving image", str(count))
                    cv2.imwrite(os.path.join(save_path, str(count) + '.png'), img)
                    count += 1
                    last_image_taken = time_now


        grabResult.Release()

    elapsed_time = time.time() - tstart
    avg_fps = (fps * duration) / elapsed_time
    print("elapsed time:", elapsed_time)
    print("avg fps:", avg_fps)

    # Releasing the resource
    camera.StopGrabbing()
    cv2.destroyAllWindows()
    if record_video:
        video_writer.release()


if __name__ == '__main__':
    save_img_path = "data/kalo_18-01-2021"
    save_video_path = "data/basler_video"

    calibration = Calibration(calibration_file="data/kalo_v2_calibration/calibration_1450x1450.yaml",
                              basler_config_file="basler_config.yaml")

    # record_from_webcam(save_img_path)
    record_from_basler(save_img_path, record_video=False, save_all_imgs=False, undistort=True, limit_fps=None)  # will save all images
    # record_from_basler(save_img_path, record_video=False, undistort=True)  # press 's' to save an image
    # record_from_basler(save_video_path, record_video=True, undistort=True)  # records to .avi file
