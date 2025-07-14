from pypylon import pylon
import cv2
import numpy as np
from utils.calibration_utils import *
from utils.comms_utils import *
from utils.analysis_utils import *
import os
import time

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
ball_search_box = [calib_data.roi_size[0]-1, calib_data.roi_size[1]-1]
ball_template = ball_template_init(calib_data.ball_box, calib_data.ball_radius)

ball_search_center_fallback = [calib_data.roi_size[0] // 2, calib_data.roi_size[1] // 2]
ball_search_box_fallback = [calib_data.roi_size[0]-1, calib_data.roi_size[1]-1]

# Kalman-Filter
kf = ball_init_kalman(calib_data.framerate, 1e-0, 1e-0)
kf = ball_init_kalman(1, 1e-2, 1e-1, 1)
ball_kalman_reinit(kf, ball_search_center, [0, 0], 1)

# Positions for ball tracking (type int)
predicted_pos = []
measured_pos = []
corrected_pos = []
last_corrected_pos = [600, 300]

predicted_pos_list = []
measured_pos_list = []
corrected_pos_list = []


# Frames
total_unknown_frames = 0
unknow_frame_numbers = []
frame_number = 0
unknown_frames = 0
reinit = True

# Times

time_stamps = []

# UDP Connection

udp_socket, udp_address = udp_init()
seq = 0

# Live-View
title_live = "Live-Feed"
cv2.namedWindow(title_live, cv2.WINDOW_NORMAL)
cv2.resizeWindow(title_live, 1600, 900)

# Fallback-Footage
rec_folder = "footage/rec05"
rec_files = sorted([
    os.path.join(rec_folder, f)
    for f in os.listdir(rec_folder)
    if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))
])

# Test for performance
if not useLive and False: # remove and False for Analysis
    for f in rec_files:
        img = cv2.imread(f)
        times = []
        times.append(time.perf_counter())

        img = dist_undistort(img, mapx, mapy, calib_data.roi_offset, calib_data.roi_size)

        times.append(time.perf_counter())

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        times.append(time.perf_counter())

        ball = ball_blur(img_hsv, calib_data.ball_blur_size)

        times.append(time.perf_counter())

        ball = ball_isolate_hsv(ball, calib_data.ball_hsv_lower, calib_data.ball_hsv_upper)
        times.append(time.perf_counter())

        ball = ball_morph(ball, ball_morph_kernel)

        times.append(time.perf_counter())

        kf_predicted = kf.predict()
        predicted_pos = [int(kf_predicted[0][0]), int(kf_predicted[1][0])]
        predicted_vel = [kf_predicted[2][0], kf_predicted[3][0]]
        #predicted_vel_pos = [int(predicted_pos[0] + predicted_vel[0] * 5), int(predicted_pos[1] + predicted_vel[1] * 5)]
        #cv2.circle(img, predicted_pos, 7, (0,0,200), 1) # Predicted Pos in Red
        #cv2.arrowedLine(img, predicted_pos, predicted_vel_pos, (200, 200, 0), 1, tipLength=0.1)
        
        if abs(predicted_vel[1]) > abs(predicted_vel[0]):
            if ball_search_box == [300, 150]:
                ball_search_box = [150, 300]

        if frame_number % 50 == 0:
            ball_search_box = ball_search_box_fallback
            ball_search_center = ball_search_center_fallback
        
        times.append(time.perf_counter())

        #cv2.rectangle(img, [ball_search_center[0] - ball_search_box[0] // 2, ball_search_center[1] - ball_search_box[1] // 2], [ball_search_center[0] + ball_search_box[0] // 2, ball_search_center[1] + ball_search_box[1] // 2], [200, 200, 200], 2)

        ball_search_center, ball_search_box, ball_state = ball_detect_template(ball, ball_template, predicted_pos, ball_search_box, 0.25)

        times.append(time.perf_counter())

        if ball_state == 1: # Ball found
            unknown_frames = 0
            ball_search_box = [300, 150] # reset serch window
            
            if reinit:
                reinit = not reinit
                ball_kalman_reinit(kf, ball_search_center, [0,0], 1)
                kf_predicted = kf.predict()
                predicted_pos = [int(kf_predicted[0][0]), int(kf_predicted[1][0])]
        #        cv2.circle(img, predicted_pos, 7, (0,200,200), 1) # Predicted Pos in Red

            # Kalman Update 
            kf_measured = np.array([[np.float32(ball_search_center[0])],[np.float32(ball_search_center[1])]], dtype=np.float32)
            kf_corrected = kf.correct(kf_measured)
            corrected_pos = [int(kf_corrected[0][0]), int(kf_corrected[1][0])]
            corrected_vel = [kf_corrected[2][0], kf_corrected[3][0]]
        #    corrected_vel_pos = [int(corrected_pos[0] + corrected_vel[0] * 5), int(corrected_pos[1] + corrected_vel[1] * 5)]
        #    cv2.circle(img, corrected_pos, 8, (200, 100, 0), 1) # Corrected Pos in Blue
        #    cv2.arrowedLine(img, corrected_pos, corrected_vel_pos, (250, 250, 100), 1, tipLength=0.1)

        elif ball_state == 0: # No sufficient match for ball
            unknown_frames += 1

            total_unknown_frames += 1
            unknow_frame_numbers.append(frame_number)

            ball_search_box[0] = min(calib_data.roi_size[0], ball_search_box[0] + unknown_frames // 2)
            ball_search_box[1] = min(calib_data.roi_size[1], ball_search_box[1] + unknown_frames // 2)

        #cv2.circle(img, ball_search_center, 12, (0,200,0), 1) # Measured Pos in Green
        
        times.append(time.perf_counter())

        player = player_blur(img_hsv, calib_data.player_blur_size)

        times.append(time.perf_counter())

        player  = player_isolate_hsv(player, calib_data.player_hsv_lower, calib_data.player_hsv_upper)

        times.append(time.perf_counter())

        player = player_morph(player, player_morph_kernel)

        times.append(time.perf_counter())
        
        groups = track_player(player)

        times.append(time.perf_counter())

        time_stamps.append(times)

        frame_number += 1
        """ cv2.imshow(title_live, img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break """
codec = [
    ["Frame-Total", 12, 0],
    ["Undistort", 1, 0],
    ["HSV-Conversion", 2, 1],
    ["Ball-Blur", 3, 2],
    ["Ball-Mask", 4, 3],
    ["Ball-Morph", 5, 4],
    ["Ball-Predict", 6, 5],
    ["Ball-Detect", 7, 6],
    ["Ball-Correct", 8, 7],
    ["Player-Blur", 9, 8],
    ["Player-Mask", 10, 9],
    ["Player-Morph", 11, 10],
    ["Player-Detect", 12, 11]
    
]
plt_title = f"Performance-Test: {rec_folder.split('/')[-1]}"
#plot_times(time_stamps, codec, plt_title, "plots")

# Test on recorded images
useLive = False
if not useLive: # remove and False for rec View
    for f in rec_files:
        img = cv2.imread(f)

        # Preprocessing
        img_hsv, img = preprocess_general(img, mapx, mapy, calib_data.roi_offset, calib_data.roi_size)

        ball = preprocess_ball(img_hsv, calib_data.ball_blur_size, calib_data.ball_hsv_lower, calib_data.ball_hsv_upper, ball_morph_kernel)
        
        # Kalman Prediction
        kf_predicted = kf.predict()
        predicted_pos = [int(kf_predicted[0][0]), int(kf_predicted[1][0])]
        predicted_vel = [kf_predicted[2][0], kf_predicted[3][0]]
        predicted_vel_pos = [int(predicted_pos[0] + predicted_vel[0] * 5), int(predicted_pos[1] + predicted_vel[1] * 5)]
        cv2.circle(img, predicted_pos, 7, (0,0,200), 1) # Predicted Pos in Red
        cv2.arrowedLine(img, predicted_pos, predicted_vel_pos, (200, 200, 0), 1, tipLength=0.1)

        # basic direction dependent seach box
        if abs(predicted_vel[1]) > abs(predicted_vel[0]):
            if ball_search_box == [300, 150]:
                ball_search_box = [150, 300]

        # big search window every 50 frames
        if frame_number % 50 == 0:
            ball_search_box = ball_search_box_fallback
            ball_search_center = ball_search_center_fallback

        cv2.rectangle(img, [ball_search_center[0] - ball_search_box[0] // 2, ball_search_center[1] - ball_search_box[1] // 2], [ball_search_center[0] + ball_search_box[0] // 2, ball_search_center[1] + ball_search_box[1] // 2], [200, 200, 200], 2)

        ball_search_center, ball_search_box, ball_state = ball_detect_template(ball, ball_template, predicted_pos, ball_search_box, 0.25)
        
        if ball_state == 1: # Ball found
            unknown_frames = 0
            ball_search_box = [300, 150] # reset serch window
            
            if reinit:
                reinit = not reinit
                ball_kalman_reinit(kf, ball_search_center, [0,0], 1)
                kf_predicted = kf.predict()
                predicted_pos = [int(kf_predicted[0][0]), int(kf_predicted[1][0])]
                cv2.circle(img, predicted_pos, 7, (0,200,200), 1) # Predicted Pos in Red

            # Kalman Update 
            kf_measured = np.array([[np.float32(ball_search_center[0])],[np.float32(ball_search_center[1])]], dtype=np.float32)
            kf_corrected = kf.correct(kf_measured)
            corrected_pos = [int(kf_corrected[0][0]), int(kf_corrected[1][0])]
            corrected_vel = [kf_corrected[2][0], kf_corrected[3][0]]
            corrected_vel_pos = [int(corrected_pos[0] + corrected_vel[0] * 5), int(corrected_pos[1] + corrected_vel[1] * 5)]
            cv2.circle(img, corrected_pos, 8, (200, 100, 0), 1) # Corrected Pos in Blue
            cv2.arrowedLine(img, corrected_pos, corrected_vel_pos, (250, 250, 100), 1, tipLength=0.1)

        elif ball_state == 0: # No sufficient match for ball
            unknown_frames += 1

            total_unknown_frames += 1
            unknow_frame_numbers.append(frame_number)

            ball_search_box[0] = min(calib_data.roi_size[0], ball_search_box[0] + unknown_frames // 2)
            ball_search_box[1] = min(calib_data.roi_size[1], ball_search_box[1] + unknown_frames // 2)

            
        cv2.circle(img, ball_search_center, 12, (0,200,0), 1) # Measured Pos in Green

        frame_number += 1
        cv2.imshow(title_live, img)
        
        key = cv2.waitKey(1) # Change to 0 for frame-by-frame
        if key == ord('n'):
            continue
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.imwrite("footage/temp_img.png", cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))

#print(total_unknown_frames)
#print(unknow_frame_numbers)

# Test on live image
if useLive:
    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab.GrabSucceeded():
            img = grab.Array

            # Preprocessing
        img_hsv, img = preprocess_general(img, mapx, mapy, calib_data.roi_offset, calib_data.roi_size)

        ball = preprocess_ball(img_hsv, calib_data.ball_blur_size, calib_data.ball_hsv_lower, calib_data.ball_hsv_upper, ball_morph_kernel)
        
        # Kalman Prediction
        kf_predicted = kf.predict()
        predicted_pos = [int(kf_predicted[0][0]), int(kf_predicted[1][0])]
        predicted_vel = [kf_predicted[2][0], kf_predicted[3][0]]
        predicted_vel_pos = [int(predicted_pos[0] + predicted_vel[0] * 5), int(predicted_pos[1] + predicted_vel[1] * 5)]
        cv2.circle(img, predicted_pos, 7, (0,0,200), 1) # Predicted Pos in Red
        cv2.arrowedLine(img, predicted_pos, predicted_vel_pos, (200, 200, 0), 1, tipLength=0.1)

        if abs(predicted_vel[1]) > abs(predicted_vel[0]):
            if ball_search_box == [300, 150]:
                ball_search_box = [150, 300]

        if frame_number % 50 == 0:
            ball_search_box = ball_search_box_fallback
            ball_search_center = ball_search_center_fallback

        cv2.rectangle(img, [ball_search_center[0] - ball_search_box[0] // 2, ball_search_center[1] - ball_search_box[1] // 2], [ball_search_center[0] + ball_search_box[0] // 2, ball_search_center[1] + ball_search_box[1] // 2], [200, 200, 200], 2)

        ball_search_center, ball_search_box, ball_state = ball_detect_template(ball, ball_template, predicted_pos, ball_search_box, 0.25)
        
        if ball_state == 1: # Ball found
            unknown_frames = 0
            ball_search_box = [300, 150] # reset serch window
            
            if reinit:
                reinit = not reinit
                ball_kalman_reinit(kf, ball_search_center, [0,0], 1)
                kf_predicted = kf.predict()
                predicted_pos = [int(kf_predicted[0][0]), int(kf_predicted[1][0])]
                cv2.circle(img, predicted_pos, 7, (0,200,200), 1) # Predicted Pos in Red

            # Kalman Update 
            kf_measured = np.array([[np.float32(ball_search_center[0])],[np.float32(ball_search_center[1])]], dtype=np.float32)
            kf_corrected = kf.correct(kf_measured)
            corrected_pos = [int(kf_corrected[0][0]), int(kf_corrected[1][0])]
            corrected_vel = [kf_corrected[2][0], kf_corrected[3][0]]
            corrected_vel_pos = [int(corrected_pos[0] + corrected_vel[0] * 5), int(corrected_pos[1] + corrected_vel[1] * 5)]
            cv2.circle(img, corrected_pos, 8, (200, 100, 0), 1) # Corrected Pos in Blue
            cv2.arrowedLine(img, corrected_pos, corrected_vel_pos, (250, 250, 100), 1, tipLength=0.1)

            # Norm and Send Player pos to Unity
            sendPos = [None, None]
            # Change -1, 1 depending on orientation in unity
            sendPos[0] = normalize_single(corrected_pos, 0, calib_data.roi_size[0], -1, 1)
            sendPos[1] = normalize_single(corrected_pos, 0, calib_data.roi_size[1], -1, 1)
            seq = udp_send_data_struct(udp_socket, udp_address, sendPos, seq, "iff")

        elif ball_state == 0: # No sufficient match for ball
            unknown_frames += 1

            total_unknown_frames += 1
            unknow_frame_numbers.append(frame_number)

            ball_search_box[0] = min(calib_data.roi_size[0], ball_search_box[0] + unknown_frames // 2)
            ball_search_box[1] = min(calib_data.roi_size[1], ball_search_box[1] + unknown_frames // 2)

            #  Norm and Send Player pos to Unity
            sendPos = [None, None]
            # Change -1, 1 depending on orientation in unity
            sendPos[0] = normalize_single(predicted_pos, 0, calib_data.roi_size[0], -1, 1)
            sendPos[1] = normalize_single(predicted_pos, 0, calib_data.roi_size[1], -1, 1)
            seq = udp_send_data_struct(udp_socket, udp_address, sendPos, seq, "iff")

        cv2.circle(img, ball_search_center, 12, (0,200,0), 1) # Measured Pos in Green
        
        

        frame_number += 1
        cv2.imshow(title_live, img)
        
        key = cv2.waitKey(1) # Change to 0 for frame-by-frame
        if key == ord('n'):
            continue
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.imwrite("footage/temp_img.png", cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))
