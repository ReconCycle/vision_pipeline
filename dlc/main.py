import cv2
import os

save_images_path = "saved_images"

def record_from_webcam(mirror=False):
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    files_in_dir = [f for f in os.listdir(save_images_path) if os.path.isfile(os.path.join(save_images_path, f))]
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
            cv2.imwrite(os.path.join(save_images_path, str(count) + '.png'), img)
            count += 1

    cv2.destroyAllWindows()



if __name__ == '__main__':
    record_from_webcam()
