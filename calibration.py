import cv2
from utils.calibration_utils import *


if __name__ == "__main__":
    calibration_path = "calibration\calibration_data.json"
    calib = load_calibration_data(calibration_path)
    img = cv2.imread("footage/rec02\Basler_daA1920-160uc__40621547__20250502_143855023_0095.bmp")
    #camera = camera_connect()
    #camera_set_parameters(camera, calib)
    
    img = cv2.imread("footage/temp_img.png")
    
    #calibrate_roi(calib, calibration_path, img, True)
    #calibrate_distortion(calib, calibration_path, "calibration\img", True)
    #calibrate_camera(calib, calibration_path)
    #calibrate_ball_mask(calib, calibration_path, img, True)
    calibrate_player_mask(calib, calibration_path, img, False)
    
