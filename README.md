# rugbycoach

## Installation
install and run openpose as instructed on their official repo (https://github.com/CMU-Perceptual-Computing-Lab/openpose#quick-start-overview) and instructions.txt. Do not just clone or download zip, download from the release tab  

## Preprocessing

Put the image or video in examples\media\. keypoints extracted by OpenPose are stored in folder output\<video name>

(Sprinting videos) Extract each frame of the video to frames\ with 
```
python extract_frames.py
```
Draw config lines on the frame with 
```
python draw_lines.py
```
after drawing lines with draw_lines.py, edit the config json file in config\ and add direction key  
~~initial test program for correct loading: map_keypoints.py~~  

## Running the detection

### Sprinting videos
```
python secondary.py
```
This script contains the detection logic and draws the keypoints. 

### Kicking videos  
Run openpose with original video! not the one processed with YOLOv8.  
Run YOLOv8 with input as video (yolov8/process_frames.py) and save=True. It will be saved as an .avi file in runs\segment\  
```
python ./yolov8/process_frames.py
```
mask of the rugby ball (sports ball class) is extracted with process_frames.py and saved to output_mask_yolo\

extract the frames from this output video to draw keypoints.
```
python extract_frames.py
```

Run the following script for results:
```
python kicking.py
```

## Visualisation
output_frames\ contains the frames with lines and keypoints drawn. The video format is stored in output_video\  