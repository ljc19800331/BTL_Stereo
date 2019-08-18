from __future__ import absolute_import, print_function, division
from pymba import *
import numpy as np
import cv2
import time

#very crude example, assumes your camera is PixelMode = BAYERRG8

# start Vimba

with Vimba() as vimba:

    # get system object
    system = vimba.getSystem()

    # list available cameras (after enabling discovery for GigE cameras)
    if system.GeVTLIsPresent:
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        time.sleep(0.2)

    cameraIds = vimba.getCameraIds()
    for cameraId in cameraIds:
        print('Camera ID:', cameraId)

    # get and open a camera
    camera0 = vimba.getCamera(cameraIds[0])
    camera0.openCamera()

    camera1 = vimba.getCamera(cameraIds[1])
    camera1.openCamera()

    # camera0.runFeatureCommand('GVSPAdjustPacketSize')

    # print(camera0)

    # list camera features
    # cameraFeatureNames = camera0.getFeatureNames()
    # for name in cameraFeatureNames:
    #     print('Camera feature:', name)

    # read info of a camera feature
    #featureInfo = camera0.getFeatureInfo('AcquisitionMode')
    #for field in featInfo.getFieldNames():
    #    print field, '--', getattr(featInfo, field)

    # get the value of a feature
    # print(camera0.AcquisitionMode)

    # set the value of a feature
    # print("The value of a feature is ", camera0.AcquisitionMode)
    # camera0.AcquisitionMode = 'Continuous'

    # create new frames for the camera
    frame0 = camera0.getFrame()    # creates a frame
    frame1 = camera1.getFrame()    # creates a second frame

    # announce frame
    frame0.announceFrame()
    frame1.announceFrame()

    flag = input("The function folder (1/2)?")

    # Define the image path
    if flag == 1:
        path_left = 'C:/Users/braintool/Documents/MATLAB/StereoVision/left_python/'
        path_right = 'C:/Users/braintool/Documents/MATLAB/StereoVision/right_python/'
    elif flag == 2:
        path_left = 'C:/Users/braintool/Documents/MATLAB/StereoVision/left_show/'
        path_right = 'C:/Users/braintool/Documents/MATLAB/StereoVision/right_show/'

    # capture a camera image
    count = 0
    Nimg = 30

    while count < Nimg:

        raw_input("press enter to continue")
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

        # get image data...
        imgData_0 = frame0.getBufferByteData()
        imgData_1 = frame1.getBufferByteData()

        moreUsefulImgData_0 = np.ndarray(buffer = frame0.getBufferByteData(),
                                       dtype = np.uint8,
                                       shape = (frame0.height,
                                                frame0.width,
                                                1))
        rgb_0 = cv2.cvtColor(moreUsefulImgData_0, cv2.COLOR_BAYER_RG2RGB)
        cv2.imwrite(path_right + str(count) + '.jpg', rgb_0)
        moreUsefulImgData_1 = np.ndarray(buffer=frame1.getBufferByteData(),
                                       dtype=np.uint8,
                                       shape=(frame1.height,
                                              frame1.width,
                                              1))
        rgb_1 = cv2.cvtColor(moreUsefulImgData_1, cv2.COLOR_BAYER_RG2RGB)
        cv2.imwrite(path_left + str(count) + '.jpg', rgb_1)
        print("image {} saved".format(count))
        count += 1
        camera0.endCapture()
        camera1.endCapture()
        time.sleep(1)

    # clean up after capture
    camera0.revokeAllFrames()
    camera1.revokeAllFrames()

    # close camera
    camera0.closeCamera()
    camera1.closeCamera()

