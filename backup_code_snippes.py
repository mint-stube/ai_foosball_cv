from pypylon import pylon
import cv2
import numpy as np
from utils.calibration_utils import *
from utils.comms_utils import *
#from utils.vision_utils import *
import os

# Load Calibration Data
calib_data = load_calibration_data("calibration\calibration_data.json")

# Folder for Fallback-Footage
footage_path = "D:/_Dateien/Studium/Projektarbeit/Code/Python/footage/"

# Connect Camera
camera = camera_connect()
if camera is not None:
    camera_set_parameters(camera, data=calib_data)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    useLive = True

else:
    print("Using static Images...")
    useLive = False


# Initialise everything

# Remapping
mapx, mapy = dist_init_maps(calib_data.camera_matrix, calib_data.distortion_coefficients, calib_data.camera_new_matrix, calib_data.camera_size)

# Kernel
ball_morph_kernel = ball_init_morph(calib_data.ball_morph_size)
player_morph_kernel = player_init_morph(calib_data.player_morph_size)

# Template-Matching
ball_search_center = [calib_data.roi_size[0] // 2, calib_data.roi_size[1] // 2]
ball_seach_box = [calib_data.roi_size[0]-1, calib_data.roi_size[1]-1]
ball_template = ball_template_init(calib_data.ball_box, calib_data.ball_radius)

# Kalman-Filter
kf = ball_init_kalman(calib_data.framerate, 1e-0, 1e-0)
ball_kalman_reinit(kf, ball_search_center, [0, 0], 1)
ball_kalman_reinit(kf, [600, 300], [0,0], 10)

last_pos = []

# Numbers
total_unknown_frames = 0
unknow_frame_numbers = []
frame_number = 0
unknown_frames = 0

# Live-View
title_live = "Live-Feed"
cv2.namedWindow(title_live, cv2.WINDOW_NORMAL)
cv2.resizeWindow(title_live, 1600, 900)

rec_folder = "footage/rec05"
rec_files = sorted([
    os.path.join(rec_folder, f)
    for f in os.listdir(rec_folder)
    if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))
])

useLive = False
if not useLive:
    for f in rec_files:

        img = cv2.imread(f)

        img_hsv, img = preprocess_general(img, mapx, mapy, calib_data.roi_offset, calib_data.roi_size)

        ball = preprocess_ball(img_hsv, calib_data.ball_blur_size, calib_data.ball_hsv_lower, calib_data.ball_hsv_upper, ball_morph_kernel)

        #img = ball
        predicted = kf.predict()

        predicted_pos = [int(predicted[0][0]),int(predicted[1][0])]
        predicted_vel = [predicted[2][0], predicted[3][0]]
        ball_search_center = predicted_pos
        cv2.circle(img, predicted_pos, calib_data.ball_radius, (0, 0, 200), 1)
        cv2.rectangle(img, [predicted_pos[0] - ball_seach_box[0] // 2, predicted_pos[1] - ball_seach_box[1] // 2 ], [predicted_pos[0] + ball_seach_box[0] // 2, predicted_pos[1] + ball_seach_box[1] // 2], (100,100, 10), 1)

        cv2.arrowedLine(img, predicted_pos, [int(predicted_pos[0] + predicted_vel[0]) * 1, int(predicted_pos[1] + predicted_vel[1]) * 1], (255,255,255), 1, tipLength=0.3)
        ball_search_center, ball_seach_box, ball_state = ball_detect_template(ball, ball_template, ball_search_center, ball_seach_box, 0.25)
        
        if ball_state == 1: # Ball found
            unknown_frames = 0
            ball_seach_box = [300, 200] # reset serch window

            measured = np.array([[np.float32(ball_search_center[0])], [np.float32(ball_search_center[1])]])
            corrected = kf.correct(measured)
            corrected_pos = [int(corrected[0][0]), int(corrected[1][0])]
            corrected_vel = [corrected[2][0], corrected[3][0]]

            cv2.circle(img, corrected_pos, calib_data.ball_radius, (200, 0, 0), 1)
            cv2.arrowedLine(img, corrected_pos, [int(corrected_pos[0] + corrected_vel[0]) * 1, int(corrected_pos[1] + corrected_vel[1]) * 1], (255,255,255), 1, tipLength=0.3)

        if ball_state == 0: # No sufficient match for ball
            unknown_frames += 1
            total_unknown_frames += 1
            unknow_frame_numbers.append(frame_number)
            ball_search_center = predicted_pos
            ball_seach_box[0] = min(calib_data.roi_size[0], ball_seach_box[0] + unknown_frames // 2)
            ball_seach_box[1] = min(calib_data.roi_size[1], ball_seach_box[1] + unknown_frames // 2)

        
            
        cv2.circle(img, ball_search_center, calib_data.ball_radius, (0,200,0), 1)
        
        """ player = preprocess_player(img_hsv, calib_data.player_blur_size, calib_data.player_hsv_lower, calib_data.player_hsv_upper, player_morph_kernel)
        groups = track_player(player)
        for group in groups:
            for g in group:
                cv2.circle(img, [g[0],g[1]], 10, (200, 100, 100), 1)
 """
        frame_number += 1
        cv2.imshow(title_live, img)
        
        key = cv2.waitKey(1) # Change to 1 for frame-by-frame
        if key == ord('n'):
            continue
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.imwrite("footage/temp_img.png", cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))

print(total_unknown_frames)
print(unknow_frame_numbers)


if useLive:
    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab.GrabSucceeded():
            img = grab.Array

            img = dist_undistort(img, mapx, mapy, calib_data.roi_offset, calib_data.roi_size)

            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            ball = ball_blur(img_hsv, calib_data.ball_blur_size)

            ball = ball_isolate_hsv(ball, calib_data.ball_hsv_lower, calib_data.ball_hsv_upper)

            ball = ball_morph(ball, ball_morph_kernel)

            #img = ball
            predicted = kf.predict()

            predicted_pos = [int(predicted[0][0]),int(predicted[1][0])]
            predicted_vel = [predicted[2][0], predicted[3][0]]
            
            if unknown_frames > 100:
                print("Lost")
                ball_search_center = [calib_data.roi_size[0] // 2, calib_data.roi_size[1] // 2]
                ball_seach_box = [calib_data.roi_size[0], calib_data.roi_size[1]]

            cv2.circle(img, predicted_pos, calib_data.ball_radius, (0, 0, 200), 1)
            cv2.rectangle(img, [predicted_pos[0] - ball_seach_box[0] // 2, predicted_pos[1] - ball_seach_box[1] // 2 ], [predicted_pos[0] + ball_seach_box[0] // 2, predicted_pos[1] + ball_seach_box[1] // 2], (100,100, 10), 1)

            #cv2.arrowedLine(img, predicted_pos, [int(predicted_pos[0] + predicted_vel[0]) * 1, int(predicted_pos[1] + predicted_vel[1]) * 1], (255,255,255), 1, tipLength=0.3)
            ball_search_center, ball_seach_box, ball_state = ball_detect_template(ball, ball_template, predicted_pos, ball_seach_box, 0.25)
            
            if ball_state == 1: # Ball found
                unknown_frames = 0
                ball_seach_box = [100, 100] # reset serch window

                measured = np.array([[np.float32(ball_search_center[0])], [np.float32(ball_search_center[1])]])
                corrected = kf.correct(measured)
                corrected_pos = [int(corrected[0][0]), int(corrected[1][0])]
                corrected_vel = [corrected[2][0], corrected[3][0]]

                cv2.circle(img, corrected_pos, calib_data.ball_radius, (200, 0, 0), 1)
                cv2.arrowedLine(img, corrected_pos, [int(corrected_pos[0] + corrected_vel[0]) * 1, int(corrected_pos[1] + corrected_vel[1]) * 1], (255,255,255), 1, tipLength=0.3)

            if ball_state == 0: # No sufficient match for ball
                unknown_frames += 1
                ball_search_center = predicted_pos
                ball_seach_box[0] = min(calib_data.roi_size[0], ball_seach_box[0] + unknown_frames // 2)
                ball_seach_box[1] = min(calib_data.roi_size[1], ball_seach_box[1] + unknown_frames // 2)

            
                
            cv2.circle(img, ball_search_center, calib_data.ball_radius, (0,200,0), 1)
            



            cv2.imshow(title_live, img)

            key = cv2.waitKey(1)
            if key == ord('n'):
                continue
            if key == ord('q'):
                break
