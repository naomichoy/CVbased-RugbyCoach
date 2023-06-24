import os
import cv2
from PIL import Image

def extract_frames(video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    count = 0

    while success:
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to PIL Image format
        pil_image = Image.fromarray(color_coverted)

        # Save the frame as an image file
        frame_path = f"{output_path}/frame_{str(count).zfill(12)}.jpg"
        # cv2.imwrite(frame_path, frame)
        pil_image.save(frame_path)

        # Read the next frame
        success, frame = video.read()
        count += 1

    # Release the video file
    video.release()


video_name = "s4.mov"       # with extension name

# Provide the path to the video file
video_path = f"examples/media/{video_name}"

# Provide the path to the output directory where the frames will be saved
output_path = f"frames/{video_name.split('.')[0]}"
print("output frames to", output_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Call the function to extract frames
extract_frames(video_path, output_path)
