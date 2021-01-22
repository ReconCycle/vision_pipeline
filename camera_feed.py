from pypylon import pylon
from image_calibration import ImageCalibration

def camera_feed(undistort=True):

    calibration = ImageCalibration(calibration_file="/home/sruiz/datasets/deeplabcut/kalo_v2_calibration/calibration_1450x1450_undistorted.yaml",
                                    basler_config_file="config/basler_config.yaml")

    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while camera.IsGrabbing():

        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()

            if undistort:
                img = calibration.undistort_imgs(img)  # resizes and undistorts image

        grabResult.Release()