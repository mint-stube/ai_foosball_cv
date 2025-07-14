# ai_foosball_cv
Github for the Computer-Vision-Part of the AI-Foosball-Project at the University of Applied Sciences Emden-Leer


FILES AND FOLDERS:

- calibration - contains images for distortion calibration as well as calibration data in .json-format  
- footage - contains folders with sequence of bitmap images  
- plots - contain plots of different measure sequences  
- selector - contains all the info and scripts for the camera and lens selection  
- utils - contains different python scripts for all the general methods used  
- UDP_Communications.cs as used in a Unity-Project for the UDP-Ping-Test  
- backup_code_snippets.py contains an old version of the tracking logic  
- calibration.py for all the calibration steps outcomment the different methods as needed  
- tracking.py for the actual tracking. Add or remove functions as needed. Consists of 3 loops, one for performance test on recorded images, one for loop over recorded images, one for live use

HOW TO INSTALL:
1. Download all files
2. install necessary packages (pylon, cv2, numpy, pandas, matplotlib)

HOW TO LIVE-TRACKING:
1. Connect Basler-Camera via USB
2. Run calibration.py with useLive set to true
3. Change framerate in calibrationdata.json
4. Adjust as needed and save changes
5. Adjust code in tracking.py as needed (you can deactivate the first 2 loops)
6. Run tracking.py

NOTES ON LIVE-TRACKING:
- disable all the visual feedback (cv2.circle, cv2.rectangle, cv2.arrow, cv2.imshow, cv2.waitkey) for better performance
- you can modify the player tracking logic to your needs

