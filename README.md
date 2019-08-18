# BTL_Stereo
This is the code for Stereo Vision using Prosilica and Logitech cameras in Brain Tool Lab, Duke University.
All the codes are created based on the following references: 
1. Pymba Python wrapper: https://github.com/morefigs/pymba
2. Python VTK: Visualization Tool Kit and example codes online
3. Python Opencv Example codes in Github

The code consists of two parts:
1. Logitech Code (using two Logitech C270 Webcam) 
2. Vimba Code (using two Prosilica GC1020 cameras), please set up the two Prosilica GC1020 cameras before using this code.

For each section of the code, the following functions are included:
0. Please change the path of the image folder at the initialization function.
1. Capture two images at one time (Calibration image pair).
2. Realtime image pair visualization (show two images in realtime with opencv tools).
3. 3D reconstruction based on StereoVision.
  a. For StereoVision_Logitech.py, the reconstruction code is included.
  b. For StereoVision_Vimba.py, the recontruction code is not included but the StereoVision_Logitech.py code can be used for this section. The user can modify the code with specific application.
4. The main function is included in each script and user scan test different functions after setting the camera hardware.

Other important notes: 
1. Calibration board and techniques (recommend to use a larger calibration board) 
  a. For Camera with small Field of View or if the object distance is relatively small ( < 30 cm), use a small calibration board
  b. For other camera setting, e.g Two logitech C270 cameras, use a large calibration board. 
2. Camera Setting 
  a. For the first time, the parallel setting is recommended for higher calibration accuracy. 
3. The 3D visualization with VTK
  a. Please refer to https://lorensen.github.io/VTKExamples/site/ for more VTK examples 
  b. The VTK functions used in this scipt are compatible with either realtime or non-realtime colorized point cloud visualization. Users are recommended to understand the VTK architecture before using this code.
4. The 3D Realtime Visualization (so far the speeding issue is not solved)
  a. The 3D colorized Realtime visualization is based on multi-thread functions. The Visualization module is running parallel with the stereo vision module. 
  b. The speeding issue is not solved if we attach the color information for the point cloud. This will be solved in future update.
