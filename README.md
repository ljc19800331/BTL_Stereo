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
1. Capture two images at one time (Calibration image pair) 
2. Realtime image pair visualization (show two images in realtime with opencv tools)
3. 3D reconstruction based on StereoVision
  a. For StereoVision_Logitech.py, the reconstruction code is included.
  b. For StereoVision_Vimba.py, the recontruction code is not included but the StereoVision_Logitech.py code can be used for this section. The user can modify the code with specific application.

Other important notes: 
1. Calibration board and techniques (recommend to use a larger calibration board) 

2. Camera Setting (recommend for parallel setting)

3. The 3D visualization with VTK

4. The 3D Realtime Visualization (so far the speeding issue is not solved)
