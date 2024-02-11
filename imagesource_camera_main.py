
"""imagings-source camera module

Draft: Guangshen Ma 
    1. simple unit testing. 
    2. generalized tutorial to start the packege.
    3. unit testing cases:

Reference: code referred from the official python code from the main company 

"""

import ctypes
import time
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import imagesource_camera_api

class imagesource_camera():

    def __init__(self):

        """initialize the camera and the robot-stage modules
            1. left-camera: 
            2. right-camera: 
        """

        # setup the camera modules 
        self.camera_obj_left = imagesource_camera_api.library_camera( cam_id = "DFK 33UP1300 1" )
        self.camera_obj_right = imagesource_camera_api.library_camera( cam_id = "DFK 33UP1300" )

    def cam_left_init(self, para_dict_left):

        # setting
        exposure_time_of_ref_img_left = para_dict_left["exposure_time_of_ref_img_left"] 
        exposure_time_of_laser_spot_img_left = para_dict_left["exposure_time_of_laser_spot_img_left"] 
        path_main = para_dict_left["path_main"] 

        if (self.camera_obj_left.cam_obj.IC_IsDevValid(self.camera_obj_left.hGrabber_obj)):
            self.camera_obj_left.cam_obj.IC_StartLive(self.camera_obj_left.hGrabber_obj, 1)
        exposure_time_of_ref_img_left = para_dict_left["exposure_time_of_ref_img_left"]
        exposure_time_of_laser_spot_img_left = para_dict_left["exposure_time_of_laser_spot_img_left"]
        self.camera_obj_left.set_exposure_time(exposure_time_of_ref_img_left)
        time.sleep(0.5)
        img_ref_left = self.camera_obj_left.get_one_image()
        self.camera_obj_left.set_exposure_time(input_exposure_time=exposure_time_of_laser_spot_img_left)
        img_of_low_exposure_left = self.camera_obj_left.get_one_image()
        
        # show the images with different exposure time
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(img_ref_left)
        plt.subplot(1, 2, 2)
        plt.imshow(img_of_low_exposure_left)
        plt.show()

        flag_save_base_img = input("save the reference image (yes or no)?")
        if flag_save_base_img == "yes":
            path_img_reference = path_main + "img_base_left.jpg"
            path_img_low_exposure = path_main + "img_low_exposure_left.jpg"
            cv2.imwrite(path_img_reference, img_ref_left)
            cv2.imwrite(path_img_low_exposure, img_of_low_exposure_left)

    def cam_right_init(self, para_dict_right):

        # setting
        exposure_time_of_ref_img_right = para_dict_right["exposure_time_of_ref_img_right"] 
        exposure_time_of_laser_spot_img_right = para_dict_right["exposure_time_of_laser_spot_img_right"] 
        path_main = para_dict_right["path_main"] 

        if (self.camera_obj_right.cam_obj.IC_IsDevValid(self.camera_obj_right.hGrabber_obj)):
            self.camera_obj_right.cam_obj.IC_StartLive(self.camera_obj_right.hGrabber_obj, 1)
        exposure_time_of_ref_img_right = para_dict_right["exposure_time_of_ref_img_right"]
        exposure_time_of_laser_spot_img_right = para_dict_right["exposure_time_of_laser_spot_img_right"]
        self.camera_obj_right.set_exposure_time(exposure_time_of_ref_img_right)
        time.sleep(0.5)
        img_ref_right = self.camera_obj_right.get_one_image()
        self.camera_obj_right.set_exposure_time(input_exposure_time=exposure_time_of_laser_spot_img_right)
        img_of_low_exposure_right = self.camera_obj_right.get_one_image()
        
        # show the images with different exposure time
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(img_ref_right)
        plt.subplot(1, 2, 2)
        plt.imshow(img_of_low_exposure_right)
        plt.show()

        flag_save_base_img = input("save the reference image (yes or no)?")
        if flag_save_base_img == "yes":
            path_img_reference = path_main + "img_base_right.jpg"
            path_img_low_exposure = path_main + "img_low_exposure_right.jpg"
            cv2.imwrite(path_img_reference, img_ref_right)
            cv2.imwrite(path_img_low_exposure, img_of_low_exposure_right)

    def test_capture_image_from_stereo_camera( self, para_stereo_input = {} ):

        """capture both images"""
        mode_cam_left = para_stereo_input["mode_cam_left"]
        mode_cam_right = para_stereo_input["mode_cam_right"]
        path_tumorid_img_left = para_stereo_input["path_tumorid_img_left"]
        path_tumorid_img_right = para_stereo_input["path_tumorid_img_right"]

        # camera image
        if mode_cam_left == "true":
            img_tmp = self.camera_obj_left.get_one_image()
            cv2.imwrite(path_tumorid_img_left + "img_left_test.jpg", img_tmp)

        if mode_cam_right == "true": 
            img_tmp = self.camera_obj_right.get_one_image()
            cv2.imwrite(path_tumorid_img_right + "img_right_test.jpg", img_tmp)

        time.sleep(0.5)

        # stop the camera
        if mode_cam_left == "true":
            self.camera_obj_left.cam_obj.IC_StopLive(self.camera_obj_left.hGrabber_obj)
            self.camera_obj_left.cam_obj.IC_ReleaseGrabber(self.camera_obj_left.hGrabber_obj)

        if mode_cam_right == "true":
            self.camera_obj_right.cam_obj.IC_StopLive(self.camera_obj_right.hGrabber_obj)
            self.camera_obj_right.cam_obj.IC_ReleaseGrabber(self.camera_obj_right.hGrabber_obj)

if __name__ == "__main__":

    # define the class
    cam_class_obj = imagesource_camera()
    cam_class_obj.test_capture_image_from_stereo_camera()
