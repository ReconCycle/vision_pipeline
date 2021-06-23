from pypylon import pylon
from image_calibration import ImageCalibration
import sys
import signal
import time


# adding this signal handler to handle CTRL+C to quit. Doesn't work sometimes otherwise
def signal_handler(signal, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def camera_feed(undistort=True, fps=None, callback=None):

    calibration = ImageCalibration()

    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    t_start = 0
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()

            if undistort:
                img = calibration.undistort_imgs(img)  # resizes and undistorts image
            
            t_now = time.time()
            if fps is None or t_now - t_start > 1/fps:
                t_start = t_now # reset t_start
                if callback:
                    callback(img)

        grabResult.Release()

if __name__ == '__main__':

    def img_from_camera(img):
        if img is not None:
            print("image received")
        else:
            print("img is none")
    
    camera_feed(callback=img_from_camera)