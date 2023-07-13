# rugbycoach

install and run openpose as instructed on their official repo and instructions.txt. Do not just clone or download zip, download from the release tab  

put the image or video in examples\media\. keypoints extracted by OpenPose are stored in folder output\<video name>

extract each frame of the video to frames\ with extract_frames.py

after drawing lines with draw_lines.py, edit the config json file in config\ and add direction key  
~~initial test program for correct loading: map_keypoints.py~~  

output_frames\ contains the frames with lines and keypoints drawn. The video format is stored in output_video\