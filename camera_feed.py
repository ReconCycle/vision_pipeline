from pypylon import pylon
from image_calibration import ImageCalibration
import sys
import signal
import time


# adding this signal handler to handle CTRL+C to quit. Doesn't work sometimes otherwise
def signal_handler(signal, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def load_userset1(camera):
    """https://docs.baslerweb.com/user-sets"""
    # Load the User Set 1 user set
    camera.UserSetSelector.SetValue(pylon.UserSetSelector_UserSet1)
    camera.UserSetLoad.Execute()

def set_width_height(camera):
    # https://docs.baslerweb.com/center-x-and-center-y
    width = 2900
    height = 2900
    camera.Width.SetValue(width)
    camera.Height.SetValue(height)

    # Center the image
    camera.CenterX.SetValue(True)
    camera.CenterY.SetValue(True)

def auto_exposure(camera):
    """https://docs.baslerweb.com/exposure-auto"""
    # Set the Exposure Auto auto function to its minimum lower limit
    # and its maximum upper limit
    minLowerLimit = camera.AutoExposureTimeLowerLimitRaw.GetMin()
    maxUpperLimit = camera.AutoExposureTimeUpperLimitRaw.GetMax()
    camera.AutoExposureTimeLowerLimitRaw.SetValue(minLowerLimit)
    camera.AutoExposureTimeUpperLimitRaw.SetValue(maxUpperLimit)
    # Set the target brightness value to 128
    camera.AutoTargetValue.SetValue(128)
    # Select auto function ROI 1
    camera.AutoFunctionAOISelector.SetValue(pylon.AutoFunctionAOISelector_AOI1)
    # Enable the 'Intensity' auto function (Gain Auto + Exposure Auto)
    # for the auto function ROI selected
    camera.AutoFunctionAOIUsageIntensity.SetValue(True)
    # Enable Exposure Auto by setting the operating mode to Continuous
    camera.ExposureAuto.SetValue(pylon.ExposureAuto_Continuous)

def auto_gain(camera):
    # Set the the Gain Auto auto function to its minimum lower limit
    # and its maximum upper limit
    minLowerLimit = camera.AutoGainRawLowerLimit.GetMin()
    maxUpperLimit = camera.AutoGainRawUpperLimit.GetMax()
    camera.AutoGainRawLowerLimit.SetValue(minLowerLimit)
    camera.AutoGainRawUpperLimit.SetValue(maxUpperLimit)
    # Specify the target value
    camera.AutoTargetValue.SetValue(150)
    # Select auto function ROI 1
    camera.AutoFunctionAOISelector.SetValue(pylon.AutoFunctionAOISelector_AOI1)
    # Enable the 'Intensity' auto function (Gain Auto + Exposure Auto)
    # for the auto function ROI selected
    camera.AutoFunctionAOIUsageIntensity.SetValue(True)
    # Enable Gain Auto by setting the operating mode to Continuous
    camera.GainAuto.SetValue(pylon.GainAuto_Continuous)

def auto_white_balance(camera):
    """https://docs.baslerweb.com/balance-white-auto"""
    # Select auto function ROI 2
    camera.AutoFunctionAOISelector.SetValue(pylon.AutoFunctionAOISelector_AOI2)
    # Enable the Balance White Auto auto function
    # for the auto function ROI selected
    camera.AutoFunctionAOIUsageWhiteBalance.SetValue(True)
    # Enable Balance White Auto by setting the operating mode to Continuous
    camera.BalanceWhiteAuto.SetValue(pylon.BalanceWhiteAuto_Continuous)


def camera_feed(undistort=True, fps=None, callback=None):

    calibration = ImageCalibration()

    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    # camera.Open() # ? do I need to do this?

    # Print the model name of the camera.
    print("Using device ", camera.GetDeviceInfo().GetModelName())

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