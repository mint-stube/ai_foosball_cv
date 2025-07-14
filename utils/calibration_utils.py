import cv2
import numpy as np
from dataclasses import dataclass
import json
from typing import List, Tuple
from .vision_utils import *
from .comms_utils import *
import os
import glob


@dataclass
class CalibrationData:
    camera_offset: Tuple[int, int]
    camera_size: Tuple[int, int]
    framerate: float
    exposure_time: float
    gain: float
    
    camera_offset_fallback: Tuple[int, int]
    camera_size_fallback: Tuple[int, int]

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: List[np.ndarray]
    translation_vectors: List[np.ndarray]
    camera_new_matrix: np.ndarray
    
    roi_offset: Tuple[int, int]
    roi_size: Tuple[int, int]

    ball_blur_size: int
    ball_hsv_lower: np.ndarray
    ball_hsv_upper: np.ndarray
    ball_morph_size: int
    ball_radius: int
    ball_box: int

    player_blur_size: int
    player_hsv_lower: np.ndarray
    player_hsv_upper: np.ndarray
    player_morph_size: int

# Load Calibration data from json file at given path and return the CalibrationData Object containing the information
def load_calibration_data(path: str):
    with open(path, 'r') as f:
        data = json.load(f)

    return CalibrationData(
        camera_offset = tuple(data["camera_offset"]),
        camera_size = tuple(data["camera_size"]),
        framerate = data["framerate"],
        exposure_time = data["exposure_time"],
        gain = data["gain"],
        camera_offset_fallback = tuple(data["camera_offset_fallback"]),
        camera_size_fallback = tuple(data["camera_size_fallback"]),
        
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64),
        distortion_coefficients = np.array(data["distortion_coefficients"], dtype=np.float64),
        rotation_vectors=[np.array(vec, dtype=np.float64) for vec in data["rotation_vectors"]],
        translation_vectors = [np.array(vec, dtype=np.float64) for vec in data["translation_vectors"]],
        camera_new_matrix = np.array(data["camera_new_matrix"], dtype=np.float64),

        roi_offset = tuple(data["roi_offset"]),
        roi_size = tuple(data["roi_size"]),
        
        ball_blur_size = data["ball_blur_size"],
        ball_hsv_lower = np.array(data["ball_hsv_lower"]),
        ball_hsv_upper = np.array(data["ball_hsv_upper"]),
        ball_morph_size = data["ball_morph_size"],
        ball_radius = data["ball_radius"],
        ball_box = data["ball_box"],
        
        player_blur_size = data["player_blur_size"],
        player_hsv_lower = np.array(data["player_hsv_lower"]),
        player_hsv_upper = np.array(data["player_hsv_upper"]),
        player_morph_size = data["player_morph_size"]
    )

# Save given CalibrationData as json at given path
def save_calibration_data(data: CalibrationData, path: str):
    data = {
        "camera_offset": list(data.camera_offset),
        "camera_size": list(data.camera_size),
        "framerate": data.framerate,
        "exposure_time": data.exposure_time,
        "gain": data.gain,
        "camera_offset_fallback": list(data.camera_offset_fallback),
        "camera_size_fallback": list(data.camera_size_fallback),
        "camera_matrix": data.camera_matrix.tolist(),
        "distortion_coefficients": data.distortion_coefficients.tolist(),
        "rotation_vectors": [vec.tolist() for vec in data.rotation_vectors],
        "translation_vectors": [vec.tolist() for vec in data.translation_vectors],
        "camera_new_matrix": data.camera_new_matrix.tolist(),
        "roi_offset": list(data.roi_offset),
        "roi_size": list(data.roi_size),
        "ball_blur_size": data.ball_blur_size,
        "ball_hsv_lower": data.ball_hsv_lower.tolist(),
        "ball_hsv_upper": data.ball_hsv_upper.tolist(),
        "ball_morph_size": data.ball_morph_size,
        "ball_radius": data.ball_radius,
        "ball_box": data.ball_box,
        "player_blur_size": data.player_blur_size,
        "player_hsv_lower": data.player_hsv_lower.tolist(),
        "player_hsv_upper": data.player_hsv_upper.tolist(),
        "player_morph_size": data.player_morph_size
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

# Split image in h, s and v channel and return images containing each channel
def hsv_split(img):
    h = img[:,:,0]
    s = img[:,:,1]
    v = img[:,:,2]
    h = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
    s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    return h, s, v

# Perform HSV-Thresholding channel wise and return complete mask as well as individual channels after thresholding
def hsv_thresh(img, lower, upper):
    h = img[:,:,0]
    s = img[:,:,1]
    v = img[:,:,2]
    h = cv2.inRange(h, int(lower[0]), int(upper[0]))
    s = cv2.inRange(s, int(lower[1]), int(upper[1]))
    v = cv2.inRange(v, int(lower[2]), int(upper[2]))
    h = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
    s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    mask = cv2.inRange(img, np.array(lower), np.array(upper))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return h, s, v, mask

# Placeholder Function for Trackbars
def nothing(_):
    pass

# Calibrate parameters for ball preprocessing on static or live image using trackbars
def calibrate_ball_mask(data: CalibrationData, path, img, useLive: bool = False):         

    # Create window for visual feedback
    title = "Ball Parameters"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1600, 800)

    h_min, s_min, v_min = data.ball_hsv_lower
    h_max, s_max, v_max = data.ball_hsv_upper

    # Create trackbars for all the parameters
    cv2.createTrackbar("h_min", title, h_min, 179, nothing)
    cv2.createTrackbar("h_max", title, h_max, 179, nothing)
    cv2.createTrackbar("s_min", title, s_min, 255, nothing)
    cv2.createTrackbar("s_max", title, s_max, 255, nothing)
    cv2.createTrackbar("v_min", title, v_min, 255, nothing)
    cv2.createTrackbar("v_max", title, v_max, 255, nothing)

    cv2.createTrackbar("blur", title, data.ball_blur_size, 13, nothing)
    cv2.createTrackbar("morph", title, data.ball_morph_size, 15, nothing)

    # Convert Fallback image to HSV
    img_hsv_base = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Try camera connection if needed
    if useLive:
        try:
            camera = camera_connect()
            camera_set_parameters(camera, data)
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        except Exception as e:
            print("Error correcting to camera, using fallback image")
            useLive = False
            pass

    while True:
        if useLive and camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                img = grab.Array
                img_hsv_base = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Get current Trackbar positions         
        h_min = cv2.getTrackbarPos("h_min", title)
        h_max = cv2.getTrackbarPos("h_max", title)
        s_min = cv2.getTrackbarPos("s_min", title)
        s_max = cv2.getTrackbarPos("s_max", title)
        v_min = cv2.getTrackbarPos("v_min", title)
        v_max = cv2.getTrackbarPos("v_max", title)

        # Make blur size uneven
        blur_size = cv2.getTrackbarPos("blur", title)
        if blur_size % 2 == 0:
            blur_size += 1

        # Make morph size uneven
        morph_size = cv2.getTrackbarPos("morph", title)
        if morph_size % 2 == 0:
            morph_size += 1

        # Generate kernel for morphological operations
        morph_kernel = ball_init_morph(morph_size)

        # blur image
        img_hsv = ball_blur(img_hsv_base, blur_size)

        # split in channels and merge for side by side
        h, s, v = hsv_split(img_hsv)
        merged_base = np.hstack([img, h, s, v])

        # generate mask for channels
        h, s, v, mask = hsv_thresh(img_hsv, [h_min, s_min, v_min], [h_max, s_max, v_max])

        # perform morphological operations on mask
        mask = ball_morph(mask, morph_kernel)

        # merge for side by side after thresholding and top - down for pre and after thresholding
        merged_mask = np.hstack([mask, h, s, v])
        res = np.vstack([merged_base, merged_mask])

        cv2.imshow(title, res)

        key = cv2.waitKey(20)
        if key == ord('q'):
            break
        if key == ord('s'):
            # save parameters
            data.ball_blur_size = blur_size
            data.ball_morph_size = morph_size
            data.ball_hsv_lower = np.array([h_min, s_min, v_min])
            data.ball_hsv_upper = np.array([h_max, s_max, v_max])
            save_calibration_data(data, path=path)
            print("Calibration saved")
    cv2.destroyAllWindows()

# Calibrate parameters for player preprocessing on static or live image using trackbars
def calibrate_player_mask(data: CalibrationData, path, img, useLive: bool = False):
    # basically same workflow as for ball
    title = "Player Parameters"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1600, 800)

    h_min, s_min, v_min = data.player_hsv_lower
    h_max, s_max, v_max = data.player_hsv_upper

    cv2.createTrackbar("h_min", title, h_min, 179, nothing)
    cv2.createTrackbar("h_max", title, h_max, 179, nothing)
    cv2.createTrackbar("s_min", title, s_min, 255, nothing)
    cv2.createTrackbar("s_max", title, s_max, 255, nothing)
    cv2.createTrackbar("v_min", title, v_min, 255, nothing)
    cv2.createTrackbar("v_max", title, v_max, 255, nothing)

    cv2.createTrackbar("blur", title, data.player_blur_size, 13, nothing)
    cv2.createTrackbar("morph", title, data.player_morph_size, 15, nothing)

    img_hsv_base = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if useLive:
        try:
            camera = camera_connect()
            camera_set_parameters(camera, data)
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        except Exception as e:
            print("Error correcting to camera, using fallback image")
            useLive = False
            pass

    while True:
        if useLive and camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                img = grab.Array
                img_hsv_base = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("h_min", title)
        h_max = cv2.getTrackbarPos("h_max", title)
        s_min = cv2.getTrackbarPos("s_min", title)
        s_max = cv2.getTrackbarPos("s_max", title)
        v_min = cv2.getTrackbarPos("v_min", title)
        v_max = cv2.getTrackbarPos("v_max", title)

        blur_size = cv2.getTrackbarPos("blur", title)
        if blur_size % 2 == 0:
            blur_size += 1

        morph_size = cv2.getTrackbarPos("morph", title)
        if morph_size % 2 == 0:
            morph_size += 1

        morph_kernel = player_init_morph(morph_size)

        img_hsv = player_blur(img_hsv_base, blur_size)

        h, s, v = hsv_split(img_hsv)
        merged_base = np.hstack([img, h, s, v])

        h, s, v, mask = hsv_thresh(img_hsv, [h_min, s_min, v_min], [h_max, s_max, v_max])

        mask = player_morph(mask, morph_kernel)

        merged_mask = np.hstack([mask, h, s, v])
        res = np.vstack([merged_base, merged_mask])

        cv2.imshow(title, res)

        key = cv2.waitKey(20)
        if key == ord('q'):
            break
        if key == ord('s'):
            data.player_blur_size = blur_size
            data.player_morph_size = morph_size
            data.player_hsv_lower = np.array([h_min, s_min, v_min])
            data.player_hsv_upper = np.array([h_max, s_max, v_max])
            save_calibration_data(data, path=path)
            print("Calibration saved")
    cv2.destroyAllWindows()

# Define Region of Interest for Camera
def calibrate_roi(data: CalibrationData, data_path, img, useLive: bool = False):

    title = "Region of Interest"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1600, 800)

    h_img, w_img = img.shape[:2]

    x_min, y_min = data.camera_offset[0], data.camera_offset[1]
    x_max, y_max = data.camera_offset[0] + data.camera_size[0], data.camera_offset[1] + data.camera_size[1]

    x_min = x_min + x_min % 2
    y_min = y_min + y_min % 2
    x_max = x_max - x_max % 2
    y_max = y_max - y_max % 2

    cv2.createTrackbar("x_min", title, x_min, data.camera_offset_fallback[0] + data.camera_size_fallback[0], nothing)
    cv2.createTrackbar("y_min", title, y_min, data.camera_offset_fallback[1] + data.camera_size_fallback[1], nothing)
    cv2.createTrackbar("x_max", title, x_max, data.camera_offset_fallback[0] + data.camera_size_fallback[0], nothing)
    cv2.createTrackbar("y_max", title, y_max, data.camera_offset_fallback[1] + data.camera_size_fallback[1], nothing)

    if useLive:
            try:
                camera = camera_connect()
                camera_set_parameters(camera, data)
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            except Exception as e:
                print (f"Error connecting to camera: {e}")
                useLive = False
            
    while True:
        # get trackbar positions and make the numbers even
        x_min = cv2.getTrackbarPos("x_min", title)
        y_min = cv2.getTrackbarPos("y_min", title)
        x_max = cv2.getTrackbarPos("x_max", title)
        y_max = cv2.getTrackbarPos("y_max", title)
        x_min = x_min + x_min % 2
        y_min = y_min + y_min % 2
        x_max = x_max - x_max % 2
        y_max = y_max - y_max % 2

        if useLive and camera.IsGrabbing():
            camera.StopGrabbing()
            camera.Width.SetValue(x_max-x_min)
            camera.Height.SetValue(y_max-y_min)
            camera.OffsetX.SetValue(x_min)
            camera.OffsetY.SetValue(y_min)
            camera.StartGrabbing()

            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                img = grab.Array
                cv2.imshow(title, img)
        
        else:
            img = img[y_min:y_max, x_min:x_max]
            cv2.imshow(title, img)
        

        key = cv2.waitKey(100)
        if key == ord('n'):
            continue

        if key == ord('q'):
            break

        if key == ord('s'):
            # save camera parameters and restart camera feed
            data.camera_offset = [x_min, y_min]
            data.camera_size = [x_max - x_min, y_max - y_min]
            save_calibration_data(data, path=data_path)
            try:
                camera.StopGrabbing()
                camera_set_parameters(camera, data)
                camera.StartGrabbing()
            except Exception as e:
                pass
            
            print("Calibration saved")
        if key == ord('r'):
            # reset parameters and roi back to fallback
            x_min, y_min = data.camera_offset_fallback[0], data.camera_offset_fallback[1]
            x_max, y_max = data.camera_offset_fallback[0] + data.camera_size_fallback[0], data.camera_offset_fallback[1] + data.camera_size_fallback[1]
            cv2.setTrackbarPos("x_min", title, x_min)
            cv2.setTrackbarPos("y_min", title, y_min)
            cv2.setTrackbarPos("x_max", title, x_max)
            cv2.setTrackbarPos("y_max", title, y_max)

            try:
                camera.StopGrabbing()
                camera.Width.SetValue(data.camera_size_fallback[0])
                camera.Height.SetValue(data.camera_size_fallback[1])
                camera.OffsetX.SetValue(data.camera_offset_fallback[0])
                camera.OffsetY.SetValue(data.camera_offset_fallback[1])
                camera.StartGrabbing()

            except Exception as e:
                print(e)

            print("Reset ROI")

    cv2.destroyAllWindows()

# Calibrate distortion parameters and camera matrizes, optionally using new test images
def calibrate_distortion(data: CalibrationData, data_path, img_path, useLive: bool = False):
    title = "Press p to save current image with checkerboard in sight"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1600, 900)

    chessboard_size = (9,7)
    frame_size = data.camera_size

    # Generate calibration images
    if useLive:
        try:
            camera = camera_connect()
            camera_set_parameters(camera, data)
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            img_index = 0
            while camera.IsGrabbing:
                grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab.GrabSucceeded():
                    img = grab.Array
                    cv2.imshow(title, img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    if key == ord('p'):
                        path = img_path + f"/calibration_img_{img_index}.png"
                        cv2.imwrite(path, img)
                        print(f"Saved as {path}")
                        img_index += 1
        except Exception as e:
            print(f"Error connecting to camera: {e}")
            print("Trying fallback images...")
        
    
    if len(os.listdir(img_path)) == 0:
        print(f"No still images in selected path: {img_path}, aborting...")
        return None

    # calculate distortion
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

    objPoints = []
    imgPoints = []

    images = glob.glob(f"{img_path}/*.png")
    for image in images:
        print(image)
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret == True:
            objPoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgPoints.append(corners)

            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(50)
        else:
            print("No Ret")
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, frame_size, None, None)
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, frame_size, 1, frame_size)
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, frame_size, 5)
    x, y, w, h = roi

    data.roi_offset = [int(x), int(y)]
    data.roi_size = [int(w), int(h)]
    #data.camera_matrix = cameraMatrix.tolist()
    #data.camera_new_matrix = newCameraMatrix.tolist()
    #data.distortion_coefficients = dist.tolist()
    #data.rotation_vectors = [rvec.tolist() for rvec in rvecs]
    #data.translation_vectors = [tvec.tolist() for tvec in tvecs]
    data.camera_matrix = cameraMatrix
    data.camera_new_matrix = newCameraMatrix
    data.distortion_coefficients = dist
    data.rotation_vectors = rvecs
    data.translation_vectors = tvecs

    title = "Undistorted image, press s to save data, press q to abort..."
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1600, 900)

    while True:
        try:
            if useLive:
                try:
                    if camera.IsGrabbing():
                        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        if grab.GrabSucceeded():
                            img = grab.Array
                except:
                    pass

            else:
                img = cv2.imread(f"{img_path}/calibration_img_0.png")
        except:
            print("Neither live camera connected nor image in folder...aborting")
            return None
        
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        dst = dst[data.roi_offset[1]:data.roi_offset[1]+data.roi_size[1],data.roi_offset[0]:data.roi_offset[0]+data.roi_size[0]]
        cv2.imshow(title, dst)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            save_calibration_data(data, data_path)
            print("Calibration data overwritten")

    cv2.destroyAllWindows()
    try:
        camera.StopGrabbing()
        camera.DestroyDevice()
    except:
        pass

# Calibrate exposure time and gain of camera on undistorted live-feed
def calibrate_camera(data: CalibrationData, data_path):
    try:
        camera = camera_connect()
        camera_set_parameters(camera, data)
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    except Exception as e:
        print("Error connecting to camera....aborting..")
        return None
    
    title = "Camera Parameters"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1600, 800)

    exposure = data.exposure_time
    gain =  data.gain

    cv2.createTrackbar("exposure_time", title, exposure, 10000, nothing)
    cv2.createTrackbar("gain", title, gain*10, 200, nothing)

    mapx, mapy = dist_init_maps(data.camera_matrix, data.distortion_coefficients, data.camera_new_matrix, data.camera_size)

    while camera.IsGrabbing:
        exposure = cv2.getTrackbarPos("exposure_time", title)
        gain = cv2.getTrackbarPos("gain", title) / 10

        camera.Gain.Value = gain
        camera.ExposureTime.Value = exposure
        
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab.GrabSucceeded():
            img = grab.Array
            img = dist_undistort(img, mapx, mapy, data.roi_offset, data.roi_size)
            cv2.imshow(title, img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                data.gain = gain
                data.exposure_time = exposure
                save_calibration_data(data, data_path)
                print("Calibration data overwritten")


# end
