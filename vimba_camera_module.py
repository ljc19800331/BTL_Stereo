

"""vimba camera application software module

Draft: Guangshen Ma 
    1. simple unit testing. 
    2. generalized tutorial to start the packege.
    3. unit testing cases:

Reference: code referred from the official python code from the main company (basic code from the offcial vimba-camera website)

"""

import cv2 
import numpy as np 
import time 
from pymba import *
import matplotlib.pyplot as plt 

class vimba_camera_module(): 

    def __init__(self): 
        self.path_camera_test = []

    def cam_vimba_init(self, exposure_time_base = 20000, exposure_time_laser = 500): 

        """set up the camera and capture an image
        """

        # initialization
        vimba = Vimba()
        vimba.startup() 
        system = vimba.getSystem()
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        camera_ids = vimba.getCameraIds()
        for cam_id in camera_ids:
            print("Camera found: ", cam_id)
        self.c0 = vimba.getCamera(camera_ids[0])
        self.c0.openCamera()
        # set the exposure time
        self.c0.ExposureTimeAbs = exposure_time_base
        try:
            self.c0.StreamBytesPerSecond = 100000000
        except:
            pass

        # camera setting
        # exposure time 
        # color channel
        # droppedframes = []
        self.c0.PixelFormat = "BGR8Packed"  
        # Creates and returns a new frame object 
        self.frame0 = self.c0.getFrame()
        # Should be called after the frame is created.
        self.frame0.announceFrame()
        # Prepare the API for incoming frames.
        self.c0.startCapture()
        # Queue frames that may be filled during frame capturing.
        self.frame0.queueFrameCapture()
        # acquisition TODO: check this function
        self.c0.runFeatureCommand("AcquisitionStart")
        # capture a quick-realtime image with an acquisition time 
        self.c0.runFeatureCommand("AcquisitionStop")
        #  Wait for a queued frame to be filled (or dequeued).
        self.frame0.waitFrameCapture(1000)
        # formulate an image
        frame_data0 = self.frame0.getBufferByteData() 
        img_base = np.ndarray(  buffer = frame_data0,
                                dtype = np.uint8,
                                shape = (self.frame0.height, self.frame0.width, self.frame0.pixel_bytes) )

        return img_base

    def cam_get_loop_img( self, para_input = {} ):

        """note: please set up the camera for initialization first
        1. step-1: setup the camera for initialization (a function) before running this code 
        2. step-2: this function requires a para_input = {} 
        """ 

        # parameter setting
        exposure_time_base = para_input["exposure_time_base"]
        exposure_time_laser = para_input["exposure_time_laser"]
        path_data_save = para_input["path_data_save"]
        flag_mode = para_input["flag_mode"]
        num_of_loop_img = para_input["num_of_loop_img"]

        # initialization 
        # set up the vimba() task 
        # camera logging
        vimba = Vimba()
        vimba.startup() 
        system = vimba.getSystem()
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        camera_ids = vimba.getCameraIds()
        for cam_id in camera_ids:
            print("Camera found: ", cam_id)
        self.c0 = vimba.getCamera(camera_ids[0])
        self.c0.openCamera()
        self.c0.ExposureTimeAbs = exposure_time_base
        try:
            self.c0.StreamBytesPerSecond = 100000000
        except:
            pass

        # camera setting
        # exposure time 
        # color channel
        # droppedframes = []
        self.c0.PixelFormat = "BGR8Packed"  
        # Creates and returns a new frame object 
        self.frame0 = self.c0.getFrame()
        # Should be called after the frame is created.
        self.frame0.announceFrame()
        # Prepare the API for incoming frames.
        self.c0.startCapture()

        # Queue frames that may be filled during frame capturing.
        self.frame0.queueFrameCapture()
        # acquisition TODO: check this function
        self.c0.runFeatureCommand("AcquisitionStart")
        # capture a quick-realtime image with an acquisition time 
        self.c0.runFeatureCommand("AcquisitionStop")
        #  Wait for a queued frame to be filled (or dequeued).
        self.frame0.waitFrameCapture(1000)
        # formulate an image
        frame_data0 = self.frame0.getBufferByteData() 
        # get the base image 
        img_base = np.ndarray(buffer=frame_data0, dtype=np.uint8, shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))

        # capture images in a loop  
        # reset the camera to a lower exposure mode
        # self.c0.ExposureTimeAbs = exposure_time_laser
        if flag_mode == "loop_img": 

            for idx_img in range(num_of_loop_img):

                # capture the image while the laser is on
                print("begin to capture the image") 

                # start the camera capture 
                self.frame0.queueFrameCapture()
                self.c0.runFeatureCommand("AcquisitionStart")
                self.c0.runFeatureCommand("AcquisitionStop")
                
                # Wait for a queued frame to be filled (or dequeued).
                self.frame0.waitFrameCapture(1000)
                
                # formulate an image
                frame_data0 = self.frame0.getBufferByteData() 
                
                # obtain the image 
                img0 = np.ndarray(buffer=frame_data0, 
                                  dtype=np.uint8, 
                                  shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
                
                # save in a better image -> fix this code
                cv2.imwrite(path_data_save + str(idx_img) + ".jpg", img0)

        print("finish the separate loop image saving")
        
        return 0 

    def cam_generalized_module(self, para_input = {}): 

        """single module to summarize all the files -> all in one module 
        1. initialize the camera module  
        2. capture a single image
        3. capture the real-time image
        4. return the camera object. 
        """

        """vimba camera"""
        # TODO: replace the camera with a new setting
        # not with the camera -> testing the configuration
        # mode_vimba_camera = "false"  
        # set the exposure time
        # exposure_time_base = 20000
        # exposure_time_laser = 500     

        # parameters as inputs 
        exposure_time_base = para_input["exposure_time_base"]
        exposure_time_laser = para_input["exposure_time_laser"]
        path_data_save = para_input["path_data_save"]
        flag_mode = para_input["flag_mode"]
        num_of_loop_img = para_input["num_of_loop_img"]

        # set up the vimba() task 
        # camera logging
        vimba = Vimba()
        vimba.startup() 
        system = vimba.getSystem()
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        camera_ids = vimba.getCameraIds()
        for cam_id in camera_ids:
            print("Camera found: ", cam_id)
        self.c0 = vimba.getCamera(camera_ids[0])
        self.c0.openCamera()
        self.c0.ExposureTimeAbs = exposure_time_base
        try:
            self.c0.StreamBytesPerSecond = 100000000
        except:
            pass

        # camera setting
        # exposure time 
        # color channel
        # droppedframes = []
        self.c0.PixelFormat = "BGR8Packed"  
        # Creates and returns a new frame object 
        self.frame0 = self.c0.getFrame()
        # Should be called after the frame is created.
        self.frame0.announceFrame()
        # Prepare the API for incoming frames.
        self.c0.startCapture()

        # Queue frames that may be filled during frame capturing.
        self.frame0.queueFrameCapture()
        # acquisition TODO: check this function
        self.c0.runFeatureCommand("AcquisitionStart")
        # capture a quick-realtime image with an acquisition time 
        self.c0.runFeatureCommand("AcquisitionStop")
        #  Wait for a queued frame to be filled (or dequeued).
        self.frame0.waitFrameCapture(1000)
        # formulate an image
        frame_data0 = self.frame0.getBufferByteData() 
        # get the base image 
        img_base = np.ndarray(buffer=frame_data0, dtype=np.uint8, shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
        
        if flag_mode == "single_base_img": 
            # vis the program
            plt.imshow(img_base)
            plt.show()
            cv2.imwrite(path_data_save + "img_base.jpg", img_base)

            return img_base

        # reset the camera to a lower exposure mode
        # self.c0.ExposureTimeAbs = exposure_time_laser

        if flag_mode == "loop_img": 

            for idx_img in range(num_of_loop_img):

                # capture the image while the laser is on
                print("begin to capture the image") 
                # start the camera capture 
                self.frame0.queueFrameCapture()
                self.c0.runFeatureCommand("AcquisitionStart")
                self.c0.runFeatureCommand("AcquisitionStop")
                #  Wait for a queued frame to be filled (or dequeued).
                self.frame0.waitFrameCapture(1000)
                # formulate an image
                frame_data0 = self.frame0.getBufferByteData() 
                # obtain the image 
                img0 = np.ndarray(buffer=frame_data0, dtype=np.uint8, shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
                # save in a better image -> fix this code
                cv2.imwrite(path_data_save + str(idx_img) + ".jpg", img0)

    def cam_img_from_two_exposure_time_mode(self, exposure_time_base = 20000, exposure_time_laser = 500 ): 

        """capture an image with two different exposure time
        1. exposure time adjusted to 100.
        2. verify again with the laser spot measurements.
        """ 
            
        # set up the vimba() task 
        # camera logging
        vimba = Vimba()
        vimba.startup() 
        system = vimba.getSystem()
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        camera_ids = vimba.getCameraIds()
        for cam_id in camera_ids:
            print("Camera found: ", cam_id)
        self.c0 = vimba.getCamera(camera_ids[0])
        self.c0.openCamera()

        # set the exposure time    
        self.c0.ExposureTimeAbs = exposure_time_base
        print("exposure_time_base = ", exposure_time_base)

        try:
            self.c0.StreamBytesPerSecond = 100000000
        except:
            pass

        # camera setting
        # exposure time 
        # color channel
        # droppedframes = []
        self.c0.PixelFormat = "BGR8Packed"  
        # Creates and returns a new frame object 
        self.frame0 = self.c0.getFrame()
        # Should be called after the frame is created.
        self.frame0.announceFrame()
        # Prepare the API for incoming frames.
        self.c0.startCapture()
        # Queue frames that may be filled during frame capturing.
        self.frame0.queueFrameCapture()
        # acquisition TODO: check this function
        self.c0.runFeatureCommand("AcquisitionStart")
        # capture a quick-realtime image with an acquisition time 
        self.c0.runFeatureCommand("AcquisitionStop")
        #  Wait for a queued frame to be filled (or dequeued).
        self.frame0.waitFrameCapture(1000)
        # formulate an image
        frame_data0 = self.frame0.getBufferByteData()

        """
        check: there is not a good solution to capture a new image with a different exposure time, 
        instead, the solution here uses to copy the previous imag
        """ 
        # get the base image
        img_exposure_base = np.ndarray( buffer=frame_data0, dtype=np.uint8, shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes) )
        img_exposure_base_copy = img_exposure_base.copy()
       
        # use a different exposure time
        time.sleep(1.0)
        # reset the camera to a lower exposure mode
        self.c0.ExposureTimeAbs = exposure_time_laser
        # capture the frame 
        self.frame0.queueFrameCapture()
        # acquisition TODO: check this function
        self.c0.runFeatureCommand("AcquisitionStart")
        # capture a quick-realtime image with an acquisition time 
        self.c0.runFeatureCommand("AcquisitionStop")
        #  Wait for a queued frame to be filled (or dequeued).
        self.frame0.waitFrameCapture(1000)
        # formulate an image
        frame_data0 = self.frame0.getBufferByteData() 
        img_exposure_laser = np.ndarray(buffer=frame_data0,
                                        dtype=np.uint8,
                                        shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Horizontally stacked subplots')
        ax1.imshow(img_exposure_base_copy)
        ax1.title.set_text('exposure base')
        ax2.imshow(img_exposure_laser)
        ax2.title.set_text('exposure laser')
        plt.show()

        exit() 

if __name__ == "__main__":

    # define the class
    cam_class_obj = vimba_camera_module()

    # # step-1: camera initialization with a single image 
    exposure_time_base = 20000
    exposure_time_laser = 500
    img_test = cam_class_obj.cam_vimba_init( exposure_time_base = exposure_time_base, exposure_time_laser = exposure_time_laser )
    plt.imshow(img_test)
    plt.show()
    exit()

    # # step-2: capture a single image
    # para_input = {} 
    # para_input["exposure_time_base"] = 20000 
    # para_input["exposure_time_laser"] = 500
    # para_input["path_data_save"] = "./data/data_tmp/"
    # para_input["flag_mode"] = "loop_img"
    # para_input["num_of_loop_img"] = 20 
    # cam_class_obj.cam_generalized_module(para_input=para_input)

    # step-3: capture a single image with different exposure time 
    # exposure_time_base = 20000
    # exposure_time_laser = 500 
    # cam_class_obj.cam_img_from_two_exposure_time_mode( exposure_time_base = exposure_time_base, exposure_time_laser = exposure_time_laser ) 