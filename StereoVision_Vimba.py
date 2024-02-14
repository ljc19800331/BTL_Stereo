'''
This script is used to explain how to use the Prosilica Camera for StereoVision
Vimba is the name of the Prosilica Camera SDK
1. Vimba python wrapper
2. Vimba single image from both cameras
3. Vimba realtime image setting (not using thread)
4. Vimba 3D reconstruction

Reference:
1. Vimba SDK
2. All the code are modified based on the pymba python wrapper
'''

from __future__ import absolute_import, print_function, division
from pymba import *
import numpy as np
import cv2
import time
import numpy as np

class StereoVimba():

    def __init__(self):
        self.CalibrationNimg = 5    # Number of calibration images
        self.Nimg = 5               # Number of images
        self.Calibration_left = './Calibration_Left_Vimba/'
        self.Calibration_right = './Calibration_Right_Vimba/'
        self.path_left_show = './Data_Vimba/Left_Show/'
        self.path_right_show = './Data_Vimba/Right_Show/'

    def GetCalibrationImg(self):

        # Get the images based on each moving state
        with Vimba() as vimba:

            # get system object
            system = vimba.getSystem()

            # list available cameras (after enabling discovery for GigE cameras)
            if system.GeVTLIsPresent:
                system.runFeatureCommand("GeVDiscoveryAllOnce")
                time.sleep(0.5) # fixed

            cameraIds = vimba.getCameraIds()
            for cameraId in cameraIds:
                print('Camera ID:', cameraId)

            # Check the calibration
            if cameraIds[0] == 'DEV_000F3100A061':
                camera0 = vimba.getCamera(cameraIds[0])     # Left Camera
                camera0.openCamera()
                camera1 = vimba.getCamera(cameraIds[1])     # Right Camera
                camera1.openCamera()
                print("The first ID is ", cameraIds[0])
                print("The second ID is ", cameraIds[1])
                print("Left camera object ", camera0)
                print("Right camera object", camera1)
            elif cameraIds[0] == 'DEV_000F3100A05F':
                camera0 = vimba.getCamera(cameraIds[1])     # Left Camera
                camera0.openCamera()
                camera1 = vimba.getCamera(cameraIds[0])     # Right Camera
                camera1.openCamera()
                print("The first ID is ", cameraIds[0])
                print("The second ID is ", cameraIds[1])
                print("Left camera object ", camera0)
                print("Right camera object", camera1)

            # create new frames for the camera
            frame0 = camera0.getFrame()    # creates a frame
            frame1 = camera1.getFrame()    # creates a second frame

            # announce frame
            frame0.announceFrame()
            frame1.announceFrame()

            # capture a camera image
            count = 0
            while count <= self.CalibrationNimg:

                camera0.startCapture()
                frame0.queueFrameCapture()
                camera0.runFeatureCommand('AcquisitionStart')
                camera0.runFeatureCommand('AcquisitionStop')
                frame0.waitFrameCapture()

                camera1.startCapture()
                frame1.queueFrameCapture()
                camera1.runFeatureCommand('AcquisitionStart')
                camera1.runFeatureCommand('AcquisitionStop')
                frame1.waitFrameCapture()

                moreUsefulImgData_0 = np.ndarray(buffer = frame0.getBufferByteData(),
                                               dtype = np.uint8,
                                               shape = (frame0.height,
                                                        frame0.width,
                                                        1))
                rgb_0 = cv2.cvtColor(moreUsefulImgData_0, cv2.COLOR_BAYER_RG2RGB)
                cv2.imwrite(self.Calibration_left + str(count) + '.jpg', rgb_0)
                moreUsefulImgData_1 = np.ndarray(buffer=frame1.getBufferByteData(),
                                               dtype=np.uint8,
                                               shape=(frame1.height,
                                                      frame1.width,
                                                      1))
                rgb_1 = cv2.cvtColor(moreUsefulImgData_1, cv2.COLOR_BAYER_RG2RGB)
                cv2.imwrite(self.Calibration_right + str(count) + '.jpg', rgb_1)
                print("image {} saved".format(count))
                count += 1
                camera0.endCapture()
                camera1.endCapture()
                time.sleep(0.5)

            # clean up after capture
            camera0.revokeAllFrames()
            camera1.revokeAllFrames()

            # close camera
            camera0.closeCamera()
            camera1.closeCamera()

    def GetReconstructionImg(self):

        # Get the images based on each moving state
        with Vimba() as vimba:

            # get system object
            system = vimba.getSystem()

            # list available cameras (after enabling discovery for GigE cameras)
            if system.GeVTLIsPresent:
                system.runFeatureCommand("GeVDiscoveryAllOnce")
                time.sleep(0.2) # fixed

            cameraIds = vimba.getCameraIds()
            for cameraId in cameraIds:
                print('Camera ID:', cameraId)

            # get and open a camera
            if cameraIds[0] == 'DEV_000F3100A061':
                camera0 = vimba.getCamera(cameraIds[0])     # Left Camera
                camera0.openCamera()
                camera1 = vimba.getCamera(cameraIds[1])     # Right Camera
                camera1.openCamera()
                print("The first ID is ", cameraIds[0])
                print("The second ID is ", cameraIds[1])
                print("Left camera object ", camera0)
                print("Right camera object", camera1)
            elif cameraIds[0] == 'DEV_000F3100A05F':
                camera0 = vimba.getCamera(cameraIds[1])  # Left Camera
                camera0.openCamera()
                camera1 = vimba.getCamera(cameraIds[0])  # Right Camera
                camera1.openCamera()
                print("The first ID is ", cameraIds[0])
                print("The second ID is ", cameraIds[1])
                print("Left camera object ", camera0)
                print("Right camera object", camera1)

            # create new frames for the camera
            frame0 = camera0.getFrame()    # creates a frame
            frame1 = camera1.getFrame()    # creates a second frame

            # announce frame
            frame0.announceFrame()
            frame1.announceFrame()

            # capture a camera image
            count = 0
            while count <= self.Nimg:

                camera0.startCapture()
                frame0.queueFrameCapture()
                camera0.runFeatureCommand('AcquisitionStart')
                camera0.runFeatureCommand('AcquisitionStop')
                frame0.waitFrameCapture()

                camera1.startCapture()
                frame1.queueFrameCapture()
                camera1.runFeatureCommand('AcquisitionStart')
                camera1.runFeatureCommand('AcquisitionStop')
                frame1.waitFrameCapture()

                moreUsefulImgData_0 = np.ndarray(buffer = frame0.getBufferByteData(),
                                               dtype = np.uint8,
                                               shape = (frame0.height,
                                                        frame0.width,
                                                        1))
                rgb_0 = cv2.cvtColor(moreUsefulImgData_0, cv2.COLOR_BAYER_RG2RGB)
                cv2.imwrite(self.path_left_show + str(count) + '.jpg', rgb_0)
                moreUsefulImgData_1 = np.ndarray(buffer=frame1.getBufferByteData(),
                                               dtype=np.uint8,
                                               shape=(frame1.height,
                                                      frame1.width,
                                                      1))
                rgb_1 = cv2.cvtColor(moreUsefulImgData_1, cv2.COLOR_BAYER_RG2RGB)
                cv2.imwrite(self.path_right_show + str(count) + '.jpg', rgb_1)
                print("image {} saved".format(count))
                count += 1
                camera0.endCapture()
                camera1.endCapture()
                time.sleep(0.5)

            # clean up after capture
            camera0.revokeAllFrames()
            camera1.revokeAllFrames()

            # close camera
            camera0.closeCamera()
            camera1.closeCamera()

    def RealtimeImg(self):

        # Test the camera in two realtime process
        with Vimba() as vimba:
            system = vimba.getSystem()

            system.runFeatureCommand("GeVDiscoveryAllOnce")
            time.sleep(0.2)

            camera_ids = vimba.getCameraIds()

            for cam_id in camera_ids:
                print("Camera found: ", cam_id)

            c0 = vimba.getCamera(camera_ids[0])
            c0.openCamera()
            c1 = vimba.getCamera(camera_ids[1])
            c1.openCamera()

            try:
                # gigE camera
                print("Packet size:", c0.GevSCPSPacketSize)
                c0.StreamBytesPerSecond = 100000000
                print("BPS:", c0.StreamBytesPerSecond)
                c1.StreamBytesPerSecond = 100000000
            except:
                # not a gigE camera
                pass

            # set pixel format
            c0.PixelFormat = "BGR8Packed"  # OPENCV DEFAULT
            time.sleep(0.2)

            c1.PixelFormat = "BGR8Packed"  # OPENCV DEFAULT
            time.sleep(0.2)

            frame0 = c0.getFrame()
            frame0.announceFrame()

            frame1 = c1.getFrame()
            frame1.announceFrame()

            c0.startCapture()
            c1.startCapture()

            framecount = 0
            droppedframes = []

            while 1:
                try:
                    frame0.queueFrameCapture()
                    frame1.queueFrameCapture()
                    success = True
                except:
                    droppedframes.append(framecount)
                    success = False
                c0.runFeatureCommand("AcquisitionStart")
                c0.runFeatureCommand("AcquisitionStop")
                frame0.waitFrameCapture(1000)
                frame_data0 = frame0.getBufferByteData()
                c1.runFeatureCommand("AcquisitionStart")
                c1.runFeatureCommand("AcquisitionStop")
                frame1.waitFrameCapture(1000)
                frame_data1 = frame1.getBufferByteData()
                if success:
                    img0 = np.ndarray(buffer=frame_data0,
                                      dtype=np.uint8,
                                      shape=(frame0.height, frame0.width, frame0.pixel_bytes))
                    cv2.imshow(str(1), img0)
                    img1 = np.ndarray(buffer=frame_data1,
                                      dtype=np.uint8,
                                      shape=(frame1.height, frame1.width, frame1.pixel_bytes))
                    cv2.imshow(str(2), img1)
                framecount += 1
                # k = cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    print("Frames displayed: %i" % framecount)
                    print("Frames dropped: %s" % droppedframes)
                    break

            c0.endCapture()
            c0.revokeAllFrames()
            c0.closeCamera()
            c1.endCapture()
            c1.revokeAllFrames()
            c1.closeCamera()

if __name__ == '__main__':

    test = StereoVimba()
    # test.GetCalibrationImg()
    # test.GetReconstructionImg()
    test.RealtimeImg()