import os
import time
import numpy as np
import json
import cv2
import math
from ultralytics import YOLO

def calculate_center(x, y, w, h):
    cx = x + (w / 2)
    cy = y + (h / 2)
    return cx, cy

video_name = "P1"     # without extension
img = './test_images/frame_000000002491.jpg'
vid = f"./test_images/{video_name}.mp4"
fps = 500
input_folder = os.path.join("test_images", video_name)
time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
test_run = False
# json_folder_path = f"output/{video_name}"
# config_file_path = f"config/{video_name}.json"
# output_video = f"output_video/{video_name}-{time_now}.avi"
# output_frames_folder = f"output_frames_yolo/{video_name}-{time_now}"
output_mask_folder = f"output_mask_yolo/{video_name}-{time_now}"

# create folders
if not os.path.exists(output_mask_folder) and not test_run:
    os.makedirs(output_mask_folder)

# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

# Load a model
model = YOLO('yolov8l-seg.pt')

# # load images as list ---------- memory error RAM not enough
# image_list = []
# for filename in os.listdir(input_folder):
#     image_path = os.path.join(input_folder, filename)
#     if os.path.isfile(image_path):
#         # image_list.append(image_path)
#         # read image with OpenCV
#         # image_cv = cv2.imread(image_path)   # only give one image output
#         results = model(vid, show=True, save=False, device=0, stream=True, classes=[0,32], conf=0.35, show_conf=True)
#         # cv2.show('frame', image_cv)
#         # cv2.imwrite(f"{output_frames_folder}/{filename}.jpg", image_cv)
#         # cv2.waitKey(3)

results = model(vid, show=True, save=True, device=0, stream=True, classes=[0, 32], conf=0.38, show_conf=True)
# Process results generator
for i, result in enumerate(results):
    print(f'frame {i}')
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Class probabilities for classification outputs

    # check class_id, sports ball is 32
    class_id = np.array(boxes.cls.cpu(), dtype="int")
    # print(class_id)

    indices = [i for i, item in enumerate(class_id) if item == 32]
    data = {}
    if len(indices) > 0:
        masks = masks.to('cpu')
        boxes = boxes.to('cpu')
        if len(indices) > 1:
            ## disregard with distance from body before kick
            # person_xywh = np.array(boxes[0].xywh.tolist())[0]
            # person_c = calculate_center(person_xywh[0], person_xywh[1], person_xywh[2], person_xywh[3])
            # ball_dist_to_person = []
            # for i, ii in enumerate(indices):
            #     ball_xywh = np.array(boxes[ii].xywh.tolist())[0]
            #     ball_c = calculate_center(ball_xywh[0], ball_xywh[1], ball_xywh[2], ball_xywh[3])
            #     distance = math.hypot(person_c[0] - ball_c[0], person_c[1] - ball_c[1])
            #     ball_dist_to_person.append(distance)
            # iind = ball_dist_to_person.index(min(ball_dist_to_person))

            ## disregard with area of ball bbox, alternatively, length of mask xy list
            ball_area_list = []
            for i, ii in enumerate(indices):
                ball_xywh = (np.array(boxes[ii].xywh.tolist())[0])
                ball_area = ball_xywh[2] * ball_xywh[3]
                ball_area_list.append(ball_area)
            iind = ball_area_list.index(max(ball_area_list))
            ind = indices[iind]
        else:
            ind = indices[0]
        # print(type(masks[ind].xy))
        # print(type(boxes[ind].xyxy))
        # print(masks[ind].xy)
        # print(boxes[ind].xyxy)
        data['mask'] = np.array(masks[ind].xy).tolist()[0]
        data['bbox'] = np.array(boxes[ind].xywh).tolist()[0]
        data['conf'] = np.array(boxes[ind].conf).tolist()[0]
    else:   # ball not detected. use previous frame?
        data['mask'] = []
        data['bbox'] = []
        data['conf'] = []

    print(data)
    if not test_run:
        output_file = f'{str(i).zfill(12)}.json'
        output_file_path = os.path.join(output_mask_folder, output_file)
        with open(output_file_path, 'w') as f:
            json.dump(data, f)
