# ai_foosball_cv
Github for the Computer-Vision-Part of the AI-Foosball-Project at the University of Applied Sciences Emden-Leer


FILES AND FOLDERS:

calibration - contains images for distortion calibration as well as calibration data in .json-format

footage - contains folders with sequence of bitmap images

plots - contain plots of different measure sequences

selector - contains all the info and scripts for the camera and lens selection

utils - contains different python scripts for all the general methods used

UDP_Communications.cs as used in a Unity-Project for the UDP-Ping-Test

backup_code_snippets.py contains an old version of the tracking logic

calibration.py for all the calibration steps outcomment the different methods as needed

tracking.py for the actual tracking. Add or remove functions as needed. Consists of 3 loops, one for performance test on recorded images, one for loop over recorded images, one for live use

