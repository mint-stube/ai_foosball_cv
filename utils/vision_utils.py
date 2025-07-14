import cv2
import numpy as np

import time
import matplotlib.pyplot as plt


#region Distortion and Conversion Real-World-Pixel-Norm

# Calculate Mapping Matrixs
def dist_init_maps(camera_matrix, dist_coefficients, new_matrix, camera_size):
    return cv2.initUndistortRectifyMap(cameraMatrix = camera_matrix, distCoeffs = dist_coefficients, R = None, newCameraMatrix = new_matrix, size = camera_size, m1type = 5)

# Perform Remapping and Cropping
def dist_undistort(img, mapx, mapy, offset, size):
    img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return img[offset[1]:offset[1]+size[1], offset[0]:offset[0]+size[0]]

# Normalize Data to Interval and round to specified digits
def normalize_single(value, in_min, in_max, out_min, out_max, precision: int = 8):
    return np.round(np.interp(value, [in_min, in_max], [out_min, out_max]), precision)

#endregion

#region Ball

# basic operations for ball
def ball_blur(img, blurSize: int):
    return cv2.GaussianBlur(img, (blurSize, blurSize), 0)

def ball_isolate_hsv(img, lower, upper):
    return cv2.inRange(img, lower, upper)

def ball_init_morph(size):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

def ball_morph(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# generate template image of given size with white circle of given radius in the center
def ball_template_init(boxSize, radius): 
    img = np.zeros((boxSize, boxSize), dtype=np.uint8)
    center = (boxSize//2, boxSize//2)
    cv2.circle(img, center, radius, 255, -1) # Generates white circle on black for Template Matching
    return img

# perform template matching on specified roi (with center and box size) within a given image
def ball_detect_template(img, template, center, box, threshold):
    img_h, img_w = img.shape
    template_center_y, template_center_x = template.shape[0] // 2, template.shape[1] // 2
    x_min = max(0, center[0] - box[0] // 2)
    y_min = max(0, center[1] - box[1] // 2)
    x_max = min(img_w, center[0] + box[0] // 2)
    y_max = min(img_h, center[1] + box[1] // 2)
    temp_roi = img[y_min : y_max, x_min : x_max] # Crops image according to center and box

    result = cv2.matchTemplate(temp_roi, template, cv2.TM_CCORR_NORMED) # Performs Template Matching on ROI with CCORR_NORMED
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    center_x = x_min + max_loc[0] + template_center_x # Recalculates Center with respect to image coordinates
    center_y = y_min + max_loc[1] + template_center_y
    
    if max_val < threshold: # Insufficient match, return old center and box and state 0
        center = center
        box = box
        state = 0
    else: # Sufficient match, return old box, new center and state 1
        center = (center_x, center_y)
        box = box
        state = 1

    return center, box, state

# initialize kalman filter and return said kalman filter object
def ball_init_kalman(framerate, init_q = 1e-2, init_r = 1e-1, init_error = 1):
    kf = cv2.KalmanFilter(4,2)

    delta_t = 1/framerate
    #delta_t = 1
    
    # transition matrix for unaccelerated movement in 2 directions
    kf.transitionMatrix = np.array([
        [1, 0, delta_t, 0],
        [0, 1, 0, delta_t],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # measurement matrix for measurement of position in 2 directions
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * init_q # Q (higher -> don't trust model)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * init_r # R (higher -> don't trust measurement)

    kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32) # Starts at pos = (0,0) and vel = (0,0)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * init_error

    return kf

# reinit given kalman filter at given position, velocity and error
def ball_kalman_reinit(kf, pos, vel, error):
    kf.statePost = np.array([[pos[0]], [pos[1]], [vel[0]], [vel[1]]], dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * error

#endregion

#region Player
# basic operations for player isolation
def player_blur(img, blurSize: int):
    return cv2.GaussianBlur(img, (blurSize, blurSize), 0)

def player_isolate_hsv(img, lower, upper):
    return cv2.inRange(img, lower, upper)

def player_init_morph(size):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

def player_morph(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# detect contours in player mask
def player_detect_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    players = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        cx = x + w // 2
        cy = y + h // 2
        players.append([cx, cy, w, h])
    
    return np.array(players)

# perform optimization on detected player contours
def track_player(img, places: np.ndarray = np.array([62, 170, 385, 600]), nums: np.ndarray = np.array([1, 2, 5, 3]), widths: int = 15, min_width: int = 10, min_distance: int = 30):
        h, w = img.shape

        players = player_detect_contours(img)
        groups = [None, None, None, None]

        # only keep contours with min width
        mask = players[:, 2] >= min_width
        players = players[mask]

        # group by rods
        cx_players = players[:,0]    
        for i, place in enumerate(places):
            mask = np.abs(cx_players - place) <= widths//2

            groups[i] = players[mask]

        # average position if two are close together
        # bring horizontal position to one value per rod
        for i in range(len(groups)):
            indices = np.argsort(groups[i][:,1])
            groups[i] = groups[i][indices]
            groups[i][:, 0] = places[i]

            if len(groups[i]) == nums[i]:
                continue
            elif len(groups[i]) == 0:
                pass
            

            elif len(groups[i]) > nums[i]:
                if i == 0:
                    groups[i] = groups[i][np.argsort(groups[i][:,2])][0]
                    continue
                indices = []
                for g in range(len(groups[i])-1):
                    if abs(groups[i][g][1]-groups[i][g+1][1]) < min_distance:
                        groups[i][g][1] += groups[i][g+1][1]-groups[i][g][1]
                        indices.append(g+1)
                groups[i] = np.delete(groups[i], indices, axis = 0)
            
        return(groups)

# perform undistortion and hsv conversion
def preprocess_general(img, mapx, mapy, roi_offset, roi_size):
    img = dist_undistort(img, mapx, mapy, roi_offset, roi_size)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV), img

# perform blurring, isolation and morph operations for ball on hsv image
def preprocess_ball(img, blur_size, hsv_lower, hsv_upper, kernel):
    img = ball_blur(img, blur_size)
    img = ball_isolate_hsv(img, hsv_lower, hsv_upper)
    return ball_morph(img, kernel)

# perform blurring, isolation and morph operations for player on hsv image
def preprocess_player(img, blur_size, hsv_lower, hsv_upper, kernel):
    img = player_blur(img, blur_size)
    img = player_isolate_hsv(img, hsv_lower, hsv_upper)
    return player_morph(img, kernel)




    


def complete_processing(img, calib_data, mapx, mapy, ball_morph_kernel):

    img = dist_undistort(img, mapx, mapy, calib_data.roi_offset, calib_data.roi_size)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ball = ball_blur(img_hsv, calib_data.ball_blur_size)

    ball = ball_isolate_hsv(ball, calib_data.ball_hsv_lower, calib_data.ball_hsv_upper)

    ball = ball_morph(ball, ball_morph_kernel)