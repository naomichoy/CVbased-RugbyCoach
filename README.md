# Computer vision-based motion analysis for high-performance rugby coaching

In this project, a computer vision-based software is implemented and tested to analyse rug-by players’ through videos for high-performance coaching. Rugby players’ movements in their training videos of a single 2D view is used for the analysis. The movement to be analysed are sprinting and kicking of the ball. Human pose estimation algorithms, in particular the OpenPose model is explored and applied in videos to extract the player’s skeletal joint points for calculations. Other deep learning models such as YOLOv8 is used to identify the location of the ball. While the system is not designed to run in real-time, performance, spatiotemporal, linear kinematic variables and angular kinematic variables computation methods are evaluated against manually annotated ground truth values. The purpose of this project is to provide coaches with data-driven insights op-timising players’ performance with simple equipment setup.


[Sprinting video](https://github.com/naomichoy/CVbased-RugbyCoach/blob/main/result_img/s2.png?raw=true)

[Kicking video](https://github.com/naomichoy/CVbased-RugbyCoach/blob/main/result_img/p2.png?raw=true)

## Installation
install and run openpose as instructed on their [official repo](https://github.com/CMU-Perceptual-Computing-Lab/openpose#quick-start-overview) and ***instructions.txt***. Do not just clone or download zip, download from the [release tab](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/tag/v1.7.0) 

The file structure of this project builds on top of the openpose repo. Copy the openpose folders into this repo.   

The rest of the packages required can be installed from requirements.txt
```
pip install -r requirements.txt
```

## Data Preprocessing

Put the image or video in examples\media\.

Extract keypoints with openpose portal demo
```
bin\OpenPoseDemo.exe --video examples\media\<video name> --write_json output\<video name>
```
Keypoints extracted by OpenPose are stored in folder output\<video name>, as one JSON file per frame

(Sprinting videos) Extract each frame of the video to frames\ with 
```
python extract_frames.py
```
Draw config lines on the frame with 
```
python draw_lines.py
```
The coordinates of the lines will be saved in the config\ directory   
Edit the config json file in config\ and add direction key  
~~initial test program for correct loading: map_keypoints.py~~  

## Running the detection

### Sprinting videos
```
python secondary.py
```
This script contains the detection logic and draws the keypoints. 

### Kicking videos  
Run openpose for keypoint detection with original video! not the one processed with YOLOv8.  

Save your YOLO model in the yolov8 directory

Run YOLOv8 with input as video (yolov8/process_frames.py) and save=True on model(). It will be saved as an .avi file in runs\segment\ 

change the model in code - https://docs.ultralytics.com/models/yolov8/#performance-metrics 
```
python ./yolov8/process_frames.py
```
mask of the rugby ball (sports ball class) is extracted with process_frames.py and saved to output_mask_yolo\

extract the frames from this output video (runs\segment\) to draw keypoints.
```
python extract_frames.py
```

Run the following script to draw keypoints and for results:
```
python kicking.py
```

## Results and Visualisation
output_frames\ contains the frames with lines and keypoints drawn.

The video format is stored in output_video\  

Numerical results are stored in logs\

## File structure
frames\ extracted frames from video input

output\ keypoints extracted from openpose

logs\ output results and debug logs

output_frames\ frames with keypoints drawn

output_mask_yolo\ yolo mask data

output_video\ final video output with keypoints drawn
