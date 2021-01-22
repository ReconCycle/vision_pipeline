from pypylon import pylon
from image_calibration import ImageCalibration

# class CameraFeed:
#     def __init__(self):
#         self.img = None

#     def camera_feed(self, undistort=True, callback=None):
#         pass    

def camera_feed(undistort=True, callback=None):

    calibration = ImageCalibration()

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

                # self.img = img

                if callback:
                    callback(img)

        grabResult.Release()