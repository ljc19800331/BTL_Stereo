'''
This code is using two Logitech cameras
This function mainly has the following functions:
1. Single image
2. Realtime Processing
3. 3D reconstruction
4. Realtime 3D Reconstruction

Referece:
1. All the code are modified based on the online coding examples
2. Opencv, VTK examples and tutorials
'''
import numpy as np
import cv2
import vtk
import time
import vtk.util.numpy_support as vtk_np
import threading

class StereoLogitech():

    def __init__(self):

        # Define the Stereo Camera Setting
        self.w = 640
        self.h = 480
        self.CamLeftID = 1
        self.CamRightID = 0

        # Define the calibration image folder
        self.calibrate_left = './Data_Logitech/Calibration_Left_Logitech/'
        self.calibrate_right = './Data_Logitech/Calibration_Right_Logitech/'

        # Define the images for 3D reconstruction
        self.left_show = './Data_Logitech/img_left.jpg'
        self.right_show = './Data_Logitech/img_right.jpg'

    def Stereo_Calib(self):

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points -- the calibration point matrix
        L_chess = 9
        W_chess = 6
        objp = np.zeros((L_chess * W_chess, 3), np.float32)
        len_grid = 17.5                                                 # measure in mm in grid
        objp[:, : 2] = np.mgrid[0 : L_chess, 0 : W_chess].T.reshape(-1, 2) * len_grid   # add 19 mm to the grid

        # Arrays to store object points and image points from all images
        objpoints = []      # 3d points in real world space
        imgpointsR = []     # 2d points in the right image plane
        imgpointsL = []     # 2d points in the left image plane

        # Call all saved images
        N_calib = 20
        NcalibImg = 0

        for i in range(N_calib):

            print("The current image is ", str(i + 1))
            t = str(i + 1)

            fname_left = self.calibrate_left + t + '.jpg'
            fname_right = self.calibrate_right + t + '.jpg'

            ChessImaL = cv2.imread(fname_left)  # Calibration_Left_Logitech side
            ChessImaR = cv2.imread(fname_right)  # Calibration_Right_Logitech side

            gray_left = cv2.cvtColor(ChessImaL, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(ChessImaR, cv2.COLOR_BGR2GRAY)

            # ret is the bool value for success
            retR, cornersR = cv2.findChessboardCorners(gray_right, (L_chess, W_chess), None)    # Define the number of chess corners at the calibration board
            retL, cornersL = cv2.findChessboardCorners(gray_left, (L_chess, W_chess), None)     # Calibration_Left_Logitech side -- define the number of chessboard corners

            if (True == retR) & (True == retL):

                objpoints.append(objp)  # Save the all the 3D world coordinates to the list -- 3D

                # Refine the corners location
                cv2.cornerSubPix(gray_right, cornersR, (11, 11), (-1, -1), criteria)
                cv2.cornerSubPix(gray_left, cornersL, (11, 11), (-1, -1), criteria)

                img_left = cv2.drawChessboardCorners(ChessImaL, (L_chess, W_chess), cornersL, retL)
                img_right = cv2.drawChessboardCorners(ChessImaR, (L_chess, W_chess), cornersR, retR)

                imgpointsR.append(cornersR)
                imgpointsL.append(cornersL)
                img_hstack = np.hstack([img_left, img_right])
                cv2.imshow('img', img_hstack)
                cv2.waitKey(50)
                NcalibImg += 1
                # else:
                #     print("The incorrect calibration idx is ", t)

        print("The number of successful calibration images are ", NcalibImg)
        cv2.destroyAllWindows()

        # exit()

        return objpoints, imgpointsL, imgpointsR, ChessImaL, ChessImaR

    def Stereo_Paras(self):

        # criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Detect the patterns from the calibration board
        objpoints, imgpointsL, imgpointsR, ChessImaL, ChessImaR = self.Stereo_Calib()

        # Get some images for the size of the image
        gray_left = cv2.cvtColor(ChessImaL, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(ChessImaR, cv2.COLOR_BGR2GRAY)

        # Determine the new values for different parameters
        # Calibration_Right_Logitech camera calibration
        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                                imgpointsR,
                                                                gray_right.shape[::-1], None, None)

        hR, wR = ChessImaR.shape[:2] # The size of the image
        OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

        # ROI used to crop the image
        # Calibration_Left_Logitech camera calibration
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                                imgpointsL,
                                                                gray_left.shape[::-1], None, None)
        hL, wL= ChessImaL.shape[:2]
        OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

        # Check the reprojection errors
        mean_error_L = 0
        tot_error_L = 0
        mean_error_R = 0
        tot_error_R = 0

        for i in range(len(objpoints)):
            imgpoints2_L, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
            error = cv2.norm(imgpointsL[i], imgpoints2_L, cv2.NORM_L2) / len(imgpoints2_L)
            tot_error_L += error

        for i in range(len(objpoints)):
            imgpoints2_R, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
            error = cv2.norm(imgpointsR[i], imgpoints2_R, cv2.NORM_L2) / len(imgpoints2_R)
            tot_error_R += error

        print("Calibration_Left_Logitech total error: ", tot_error_L)
        print("The total number of left points ", len(objpoints))
        print("Calibration_Left_Logitech mean error: ", tot_error_L / len(objpoints))
        print("Calibration_Right_Logitech total error: ", tot_error_R)
        print("The total number of right points ", len(objpoints))
        print("Calibration_Right_Logitech mean error: ", tot_error_R / len(objpoints))

        # Stereo calibrate function
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        retS = None
        MLS = None
        dLS = None
        dRS = None
        R = None
        T = None
        E = None
        F = None

        retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                  imgpointsL,
                                                                  imgpointsR,
                                                                  mtxL,
                                                                  distL,
                                                                  mtxR,
                                                                  distR,
                                                                  gray_right.shape[::-1],
                                                                  flags = cv2.CALIB_FIX_INTRINSIC)
                                                                  # criteria_stereo,
                                                                  # flags)

        print("The translation between the first and second camera is ", T)

        '''
        Stereo Rectification Process
        '''
        # StereoRectify function
        # last paramater is alpha, if 0= croped, if 1= not croped
        rectify_scale = 0 # if 0 image croped, if 1 image nor croped
        RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, gray_right.shape[::-1], R, T, rectify_scale, (0,0))

        # print("The Q matrix is", Q)

        # initUndistortRectifyMap function -- map the images to the undistorted images
        # cv2.CV_16SC2 this format enables us the programme to work faster
        Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, gray_left.shape[::-1], cv2.CV_16SC2)
        Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, gray_right.shape[::-1], cv2.CV_16SC2)

        return Left_Stereo_Map, Right_Stereo_Map, Q

    def Disparity_Map(self, range_disparity):

        # Filtering
        kernel = np.ones((3, 3), np.uint8)

        # Create StereoSGBM and prepare all parameters
        window_size = 5
        min_disp = 16
        num_disp = range_disparity - min_disp
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                       numDisparities = num_disp,
                                       blockSize = window_size,
                                       uniquenessRatio = 10,
                                       speckleWindowSize = 100,
                                       speckleRange = 32,
                                       disp12MaxDiff = 5,
                                       P1 = 8*3*window_size**2,
                                       P2 = 32*3*window_size**2)

        # Used for the filtered image
        stereoR = cv2.ximgproc.createRightMatcher(stereo)   # Create another stereo for right this time

        # WLS FILTER Parameters
        lmbda = 80000
        sigma = 1.8
        visual_multiplier = 1.0
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = stereo)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        return stereo, stereoR, wls_filter

    def StereoSingle(self):

        # Save single image pair to the specific folder
        # Call the two cameras -- This is the left and right camera ID
        camL = cv2.VideoCapture(self.CamLeftID)
        camR = cv2.VideoCapture(self.CamRightID)

        try:
            time.sleep(0.5)
            raw_input("Save one image pair ?")

            # Original image from both cameras
            retR, frameR = camR.read()
            retL, frameL = camL.read()

            img_left = frameL
            img_right = frameR
            cv2.imwrite('img_left_test.jpg', img_left)
            cv2.imwrite('img_right_test.jpg', img_right)

        except:
            camR.release()
            camL.release()
            cv2.destroyAllWindows()

    def StereoShow(self):

        # Define the disparity range
        kernel = np.ones((3, 3), np.uint8)
        max_disp = 192          # user define
        min_disp = 16           # minimal disparity
        num_disp = max_disp - min_disp

        # Stereo vision camera calibration after rectification
        Left_Stereo_Map, Right_Stereo_Map, Q = self.Stereo_Paras()
        # print("The Q matrix from stereo rectify", Q)

        # Disparity map operator
        stereo, stereoR, wls_filter = self.Disparity_Map(max_disp)

        # Read teh left and right images in RGB channel
        frameL = cv2.imread(self.left_show)
        frameR = cv2.imread(self.right_show)

        # Rectify the image: Map the distored image to the undistored image
        Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # The new stereo vision image -- input for the stereo vision matching -- this is important
        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)

        # Compute the 2 images for the Depth_image
        disp = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0

        # Disparity matching and 3D point cloud
        disp_use = disp
        points = cv2.reprojectImageTo3D(disp_use, Q)
        colors = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
        mask = disp_use > disp_use.min()
        out_points = points[mask]
        out_colors = colors[mask]
        pc = out_points.reshape(-1, 3)
        colors = out_colors.reshape(-1, 3)

        # Remove the invalid values
        # mask_invalid = ((disp_use > disp_use.min()) & np.all(~np.isnan(points), axis=1) & np.all(~np.isinf(points), axis=1))

        # Denoise the data -- remove the invalid data points
        Lx = np.isfinite(pc[:, 0])
        Ly = np.isfinite(pc[:, 1])
        Lz = np.isfinite(pc[:, 2])
        L_use = Lx * Ly * Lz
        pc_use = pc[L_use, :]
        colors_use = colors[L_use, :]

        # point cloud post processing -- define the useful point cloud range (this is important)
        ScaleFactor = 1000  # convert mm to m
        pc_show = pc_use / ScaleFactor

        # Define the point cloud range -- define the 3D ROI region
        x1 = -1
        x2 = 1
        y1 = -1
        y2 = 1
        z1 = 0
        z2 = 1
        idx_roi = np.where((pc_show[:, 0] > x1)) and np.where((pc_show[:, 0] < x2)) and np.where(
                           (pc_show[:, 1] > y1)) and np.where((pc_show[:, 1] < y2)) and np.where(
                           (pc_show[:, 2] > z1)) and np.where((pc_show[:, 2] < z2))

        # The final point cloud for visualization
        pc_final = pc_show[idx_roi[0], :]
        color_final = colors_use[idx_roi[0], :]

        # Show the colorized point cloud with VTK object
        actor_color = ActorNpyColor(pc_final, color_final)
        VizActor([actor_color])

    def StereoRealtime(self):

        # Call the two cameras -- This is the left and right camera ID
        camL = cv2.VideoCapture(self.CamLeftID)
        camR = cv2.VideoCapture(self.CamRightID)

        # Calculate the Stereo Operator
        kernel = np.ones((3, 3), np.uint8)
        min_disp = 16
        max_disp = 192
        num_disp = max_disp - min_disp

        # Stereo Rectification (calibration process)
        Left_Stereo_Map, Right_Stereo_Map, Q = self.Stereo_Paras()

        # Disparity map operator
        stereo, stereoR, wls_filter = self.Disparity_Map(max_disp)

        while True:

            # Original image from both cameras
            retR, frameR = camR.read()
            retL, frameL = camL.read()

            # Rectified image
            Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

            # Grayscale image
            # grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
            # grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

            # Compute the 2 images for the Depth_image
            disp = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0
            points = cv2.reprojectImageTo3D(disp, Q)
            colors = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
            # mask = disp > disp.min()
            # out_points = points[mask]
            # out_colors = colors[mask]
            # pc = out_points.reshape(-1, 3)
            # colors = out_colors.reshape(-1, 3)
            pc = points.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            self.pc = pc
            self.colors = colors
            print(self.pc.shape)
            print(self.colors.shape)
            cv2.imshow('Disparity Mathcing', disp.astype(np.uint8))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camR.release()
        camL.release()
        cv2.destroyAllWindows()

    def StereoVTKRealtime(self):

        # Show the 3D vessel contour in realtime
        update_on = threading.Event()
        update_on.set()
        threadLock = threading.Lock()

        # ActorWrapper
        pc = np.ascontiguousarray(np.zeros((self.w * self.h, 3)))
        actorWrapper = VTKActorWrapper(pc)
        actorWrapper.update(threadLock, update_on)

        # Visualization
        viz = VTKVisualisation(threadLock, actorWrapper)
        viz.iren.Start()
        update_on.clear()

class VTKActorWrapper():

    def __init__(self, nparray):

        # Define the camera ID and parameters
        self.CamLeftID = 1
        self.CamRightID = 0
        self.w = 640
        self.h = 480

        # Initialization for visualization
        self.nparray = nparray
        nCoords = nparray.shape[0]
        nElem = nparray.shape[1]
        self.verts = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        # self.scalars = vtk.vtkUnsignedCharArray()

        # Define the color object
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")

        self.pd = vtk.vtkPolyData()
        self.verts.SetData(vtk_np.numpy_to_vtk(np.real(np.asarray(nparray))))
        self.cells_npy = np.vstack([np.ones(nCoords, dtype=np.int64), np.arange(nCoords, dtype=np.int64)]).T.flatten()
        self.cells.SetCells(nCoords, vtk_np.numpy_to_vtkIdTypeArray(self.cells_npy))
        self.pd.SetPoints(self.verts)
        self.pd.SetVerts(self.cells)
        # self.pd.GetPointData().SetScalars(self.scalars)
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputDataObject(self.pd)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetRepresentationToPoints()
        self.actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.actor.GetProperty().SetOpacity(0.5)
        self.actor.GetProperty().SetPointSize(3.0)

    def update(self, threadLock, update_on):
        thread = threading.Thread(target=self.update_actor, args=(threadLock, update_on))
        thread.start()

    def update_actor(self, threadLock, update_on):

        # Define some basic parameters
        self.w = 640
        self.h = 480
        self.pc_base = np.zeros((self.w * self.h, 3))
        self.t1 = 0
        self.t2 = len(self.pc_base)
        self.count = 0              # The initial index

        # Define Stereo Object
        StereoObj = StereoLogitech()

        # Call the two cameras -- This is the left and right camera ID
        camL = cv2.VideoCapture(self.CamLeftID)
        camR = cv2.VideoCapture(self.CamRightID)

        # Calculate the Stereo Operator
        kernel = np.ones((3, 3), np.uint8)
        min_disp = 16
        max_disp = 192
        num_disp = max_disp - min_disp

        # Stereo Rectification (calibration process)
        Left_Stereo_Map, Right_Stereo_Map, Q = StereoObj.Stereo_Paras()

        # Disparity map operator
        stereo, stereoR, wls_filter = StereoObj.Disparity_Map(max_disp)

        while True:

            time.sleep(0.001)
            threadLock.acquire()

            # Obtain the points and colors
            retR, frameR = camR.read()
            retL, frameL = camL.read()

            # Rectified image
            Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4,
                                  cv2.BORDER_CONSTANT, 0)
            Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                                   cv2.BORDER_CONSTANT, 0)

            # Compute the 2 images for the Depth_image
            disp = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0
            points = cv2.reprojectImageTo3D(disp, Q)
            colors = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
            # mask = disp > disp.min()
            # out_points = points[mask]
            # out_colors = colors[mask]
            pc = points.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            ScaleFactor = 1000  # mm to m
            pc_show = pc / ScaleFactor
            # print(pc.shape)
            # print(colors.shape)
            # exit()

            # Update the color object -- the delay is so large
            t1 = time.clock()
            for i in range(len(colors)):
                self.Colors.InsertNextTuple3(colors[i][0], colors[i][1], colors[i][2])
            t2 = time.clock()
            print(t2 - t1)

            # Update the points
            self.nparray[:] = pc_show
            self.pd.GetPointData().SetScalars(self.Colors)
            self.verts.Modified()
            self.cells.Modified()
            self.pd.Modified()

            threadLock.release()
            cv2.imshow('Disparity Mathcing', disp.astype(np.uint8))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camR.release()
        camL.release()
        cv2.destroyAllWindows()

class VTKVisualisation():

    def __init__(self, threadLock, actorWrapper_1, axis=True, ):

        self.threadLock = threadLock
        self.ren = vtk.vtkRenderer()
        self.transform = vtk.vtkTransform()
        self.transform.Translate(0.0, 0.0, 0.0)
        self.axesActor = vtk.vtkAxesActor()
        self.axesActor.SetUserTransform(self.transform)
        self.axesActor.AxisLabelsOff()
        self.axesActor.SetTotalLength(1, 1, 1)
        self.ren.AddActor(self.axesActor)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

        # pending in this case
        self.ren.AddActor(actorWrapper_1.actor)
        self.renWin.Render()
        self.iren.Initialize()
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)
        self.iren.AddObserver("TimerEvent", self.update_visualisation)
        dt = 30                 # ms -- fps
        timer_id = self.iren.CreateRepeatingTimer(dt)

    def update_visualisation(self, obj = None, event = None):
        time.sleep(0.01)
        time_start = time.clock()
        self.threadLock.acquire()
        self.ren.GetRenderWindow().Render()
        self.threadLock.release()
        time_elapsed = (time.clock() - time_start)
        # print(time_elapsed)

def VizActor(actor_list):

    transform = vtk.vtkTransform()      # transformation of a 3D axis
    transform.Translate(0.0, 0.0, 0.0)  # Remain the default setting
    axes = vtk.vtkAxesActor()           # Add the axis actor
    axes.SetUserTransform(transform)

    # Renderer -- with a loop
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(.2, .3, .4)  # Set background color
    renderer.ResetCamera()
    renderer.AddActor(axes)
    for item in actor_list:
        renderer.AddActor(item)

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Begin Interaction
    renderWindow.Render()
    renderWindowInteractor.Start()

def ActorNpyColor(npy_data, vec_color):

    # input: npy matrix and the color vector
    # output: actor

    # from npy to array
    x = npy_data[:, 0]
    y = npy_data[:, 1]
    z = npy_data[:, 2]

    # Set up the point and vertices
    Points = vtk.vtkPoints()
    Vertices = vtk.vtkCellArray()

    # Set up the color objects
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    length = int(len(x))

    # Set up the point and vertice object
    for i in range(length):
        p_x = x[i]
        p_y = y[i]
        p_z = z[i]
        id = Points.InsertNextPoint(p_x, p_y, p_z)
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(id)
        if len(vec_color) > 3: # a color vector
            Colors.InsertNextTuple3(vec_color[i][0], vec_color[i][1], vec_color[i][2])
        else:
            Colors.InsertNextTuple3(vec_color[0], vec_color[1], vec_color[2])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)
    polydata.SetVerts(Vertices)
    polydata.GetPointData().SetScalars(Colors)  # Set the color points for the problem
    polydata.Modified()

    # Set up the actor and mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

if __name__ == "__main__":

    test = StereoLogitech()
    # test.StereoSingle()
    # test.StereoVTKRealtime()
    # test.StereoRealtime()
    test.StereoShow()