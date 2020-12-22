import cv2
import os
import regex as re
import numpy as np
import shutil


def convert_folders_to_avi(path):
    # format cam0, ..., cam14 image folders to avi files.
    for camera in np.arange(15):
        cam = "cam" + str(camera)

        image_folder = os.path.join(path, 'labeled-data/' + cam)
        video_name = os.path.join(path, 'videos/' + cam + '.avi')

        print(image_folder)

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort(key=lambda f: int(re.sub('\D', '', f)))
        print("images", images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width, height))
        # video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        # cv2.destroyAllWindows()
        video.release()


def convert_folder_to_avi_batches(image_folder, video_folder, batch_size=10, fps=1.0, starting_value=0):

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort(key=lambda f: int(re.sub('\D', '', f)))
        print("images", images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = None
        for i in range(len(images)):
            print("i", i)
            if i % batch_size == 0:
                video_name = os.path.join(video_folder, str(int((i/batch_size) + starting_value)) + '.avi')
                print("video name", video_name)
                if video is not None:
                    video.release()
                video = cv2.VideoWriter(video_name, 0, fps, (width, height))

            # write the frame
            video.write(cv2.imread(os.path.join(image_folder, images[i])))

        # cv2.destroyAllWindows()
        video.release()


def copy_imgs_into_batches(image_folder, output_folder, batch_size=10, starting_value=0):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda f: int(re.sub('\D', '', f)))
    images = [os.path.join(image_folder, image) for image in images]

    current_folder_path = None
    for i in range(len(images)):
        print("i", i)
        if i % batch_size == 0:
            # make folder
            current_folder_path = os.path.join(output_folder, str(int((i // batch_size) + starting_value)))
            os.makedirs(current_folder_path)

        shutil.copy(images[i], os.path.join(current_folder_path, "img" + str(i % batch_size) + ".png"))


if __name__ == '__main__':
    image_folder = 'data/video_20-11-2020-labeled'
    video_folder = 'data/video_20-11-2020-labeled'
    convert_folder_to_avi_batches(image_folder, video_folder, batch_size=624, fps=1.5, starting_value=0)
    # copy_imgs_into_batches(image_folder, image_folder, starting_value=0)
