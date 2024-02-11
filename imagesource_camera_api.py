"""
Draft: Guangshen Ma, Ph.D.

"""

import ctypes
import time

import sys
sys.path.append("./hardware/IC-Imaging-Control-Samples-master/Python/tisgrabber/samples")
import tisgrabber as tis

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class library_camera():

    def __init__(self,
                 cam_id = "DFK 33UP1300",
                 path_cam_dll_ref = "./hardware/IC-Imaging-Control-Samples-master/Python/tisgrabber/samples/tisgrabber_x64.dll",
                 cam_img_config = "RGB32 (640x480)",
                 cam_img_frame_rate = 30.0):

        """initialize an camera object
        cam_id: the id of the camera
        path_cam_dll_ref: reference path of the .dll file, please selcet the optimal reference path before starting
        cam_img_config: very basic camera configuration for the image format, such as RGB and image shape.
        cam_img_frame_rate: frame per second
        """

        # load the linker file
        self.cam_obj = ctypes.cdll.LoadLibrary(path_cam_dll_ref)
        tis.declareFunctions(self.cam_obj)
        self.cam_obj.IC_InitLibrary(0)
        self.hGrabber_obj = self.cam_obj.IC_CreateGrabber()
        self.cam_obj.IC_OpenVideoCaptureDevice(self.hGrabber_obj, tis.T(cam_id))
        self.cam_obj.IC_SetVideoFormat(self.hGrabber_obj, tis.T(cam_img_config))
        self.cam_obj.IC_SetFrameRate(self.hGrabber_obj, ctypes.c_float(cam_img_frame_rate))
        time.sleep(1.0)

        # set exposure time
        # self.exposure_time_set = 0.0001
        # self.exposure_time_set = 0.02
        # exposure_time_set = 0.02
        # expmin = ctypes.c_float()
        # expmax = ctypes.c_float()
        # exposure = ctypes.c_float()
        # # self.cam_obj.IC_SetPropertySwitch(self.hGrabber_obj, tis.T("Exposure"), tis.T("Auto"), 0)
        # self.cam_obj.IC_SetPropertyAbsoluteValue(self.hGrabber_obj, tis.T("Exposure"), tis.T("Value"), ctypes.c_float(self.exposure_time_set))
        # self.cam_obj.IC_GetPropertyAbsoluteValue(self.hGrabber_obj, tis.T("Exposure"), tis.T("Value"), exposure)
        # self.cam_obj.IC_GetPropertyAbsoluteValueRange(self.hGrabber_obj, tis.T("Exposure"), tis.T("Value"), expmin, expmax)
        # print("Exposure is {0}, range is {1} - {2}".format(exposure.value, expmin.value, expmax.value))

    def test_exposure_time_setting(self):
        """test the setting of the exposure time setting"""

        # set the exposure time
        self.set_exposure_time(input_exposure_time=0.01)

        # initialize the camera
        if (self.cam_obj.IC_IsDevValid(self.hGrabber_obj)):
            self.cam_obj.IC_StartLive(self.hGrabber_obj, 1)
        time.sleep(1.0)

        # show the image
        img_tmp = self.get_one_image()
        plt.imshow(img_tmp)
        plt.show()

        # reset the exposure time
        self.set_exposure_time(input_exposure_time=0.0001)

        # show the image
        img_tmp = self.get_one_image()
        plt.imshow(img_tmp)
        plt.show()

        self.cam_obj.IC_StopLive(self.hGrabber_obj)
        time.sleep(0.5)
        self.cam_obj.IC_ReleaseGrabber(self.hGrabber_obj)

        print("finish the exposure time testing")

    def set_exposure_time(self, input_exposure_time):

        expmin = ctypes.c_float()
        expmax = ctypes.c_float()
        exposure = ctypes.c_float()
        # self.cam_obj.IC_SetPropertySwitch(self.hGrabber_obj, tis.T("Exposure"), tis.T("Auto"), 0)
        self.cam_obj.IC_SetPropertyAbsoluteValue(self.hGrabber_obj, tis.T("Exposure"), tis.T("Value"), ctypes.c_float(input_exposure_time))
        self.cam_obj.IC_GetPropertyAbsoluteValue(self.hGrabber_obj, tis.T("Exposure"), tis.T("Value"), exposure)
        self.cam_obj.IC_GetPropertyAbsoluteValueRange(self.hGrabber_obj, tis.T("Exposure"), tis.T("Value"), expmin, expmax)
        print("Exposure is {0}, range is {1} - {2}".format(exposure.value, expmin.value, expmax.value))

        time.sleep(1.0)

    def live_capture_img(self):
        """capture the image from a live mode
        the camera start live streaming
        the camera capture each image buffer in real-time, but not the image object.
        the image object (after capture) is converted to the image .jpg format.
        the .jpg format can be manipulated by the opencv library.
        """

        if (self.cam_obj.IC_IsDevValid(self.hGrabber_obj)):
            self.cam_obj.IC_StartLive(self.hGrabber_obj, 1)
            idx_img = 1
            idx_stop = 60
            while(idx_img < idx_stop):

                # show the real-time images
                if self.cam_obj.IC_SnapImage(self.hGrabber_obj, 2000) == tis.IC_SUCCESS:

                    # Declare variables of image description
                    Width = ctypes.c_long()
                    Height = ctypes.c_long()
                    BitsPerPixel = ctypes.c_int()
                    colorformat = ctypes.c_int()

                    # Query the values of image description
                    self.cam_obj.IC_GetImageDescription(self.hGrabber_obj, Width, Height,
                                              BitsPerPixel, colorformat)

                    # Calculate the buffer size
                    bpp = int(BitsPerPixel.value / 8.0)
                    buffer_size = Width.value * Height.value * BitsPerPixel.value

                    # Get the image data
                    imagePtr = self.cam_obj.IC_GetImagePtr(self.hGrabber_obj)

                    imagedata = ctypes.cast(imagePtr, ctypes.POINTER(ctypes.c_ubyte *
                                                           buffer_size))

                    # Create the numpy array
                    image = np.ndarray(buffer=imagedata.contents,
                                       dtype=np.uint8,
                                       shape=(Height.value,
                                              Width.value,
                                              bpp))
                    img_vis = cv2.flip(image, 0)

                    cv2.imshow('Window', img_vis)
                    cv2.waitKey(50)
                    # cv2.imwrite(self.path_img_tmp + str(idx_img) + ".jpg", image)

                time.sleep(0.01)
                print("idx_img = ", idx_img)
                idx_img += 1

            # ic.IC_StopLive(hGrabber)
        else:
            self.cam_obj.IC_MsgBox(tis.T("No device opened"), tis.T("Simple Live Video"))

        self.cam_obj.IC_StopLive(self.hGrabber_obj)
        time.sleep(0.5)
        self.cam_obj.IC_ReleaseGrabber(self.hGrabber_obj)

    def get_one_image(self):

        # show the real-time images
        if self.cam_obj.IC_SnapImage(self.hGrabber_obj, 2000) == tis.IC_SUCCESS:
            # Declare variables of image description
            Width = ctypes.c_long()
            Height = ctypes.c_long()
            BitsPerPixel = ctypes.c_int()
            colorformat = ctypes.c_int()

            # Query the values of image description
            self.cam_obj.IC_GetImageDescription(self.hGrabber_obj, Width, Height, BitsPerPixel,
                                                           colorformat)

            # Calculate the buffer size
            bpp = int(BitsPerPixel.value / 8.0)
            buffer_size = Width.value * Height.value * BitsPerPixel.value

            # Get the image data
            imagePtr = self.cam_obj.IC_GetImagePtr(self.hGrabber_obj)

            imagedata = ctypes.cast(imagePtr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))

            # Create the numpy array
            image = np.ndarray(buffer=imagedata.contents, dtype=np.uint8, shape=(Height.value, Width.value, bpp))

            img_use = cv2.flip(image, 0)

        return img_use

    def save_one_img(self, path_save_one_img = []):
        """save the image in a static way
        """

        # path_save_one_img = self.path_img_tmp
        idx_tmp = 0

        if (self.cam_obj.IC_IsDevValid(self.hGrabber_obj)):
            self.cam_obj.IC_StartLive(self.hGrabber_obj, 1)
            key = ""
            while key != "q":
                print("s: Save an image")
                print("q: End program")
                key = input('Enter your choice:')
                if key == "s":
                    if self.cam_obj.IC_SnapImage(self.hGrabber_obj, 2000) == tis.IC_SUCCESS:
                        path_img_tmp = path_save_one_img + str(idx_tmp) + ".jpg"
                        self.cam_obj.IC_SaveImage(self.hGrabber_obj, tis.T(path_img_tmp), tis.ImageFileTypes['JPEG'], 90)
                        print("Image saved.")
                        idx_tmp += 1
                    else:
                        print("No frame received in 2 seconds.")
            self.cam_obj.IC_StopLive(self.hGrabber_obj)
        else:
            self.cam_obj.IC_MsgBox(tis.T("No device opened"), tis.T("Simple Live Video"))

if __name__ == "__main__":

    # # test-1: stage movements
    # print(os.getcwd())
    cam_setting_path = "./hardware/IC-Imaging-Control-Samples-master/Python/tisgrabber/samples/tisgrabber_x64.dll"
    class_test = library_camera(path_cam_dll_ref=cam_setting_path)
    print("camera loaded successfully")

    # save single image in a loop
    path_save_one_img = []
    class_test.save_one_img()

    # test the live streaming mode
    # class_test.live_capture_img()

    # test the exposure time function
    # class_test.test_exposure_time_setting()
