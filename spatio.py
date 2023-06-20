import os
import re
import json
import cv2
from PIL import Image
import numpy as np

def read_json_file(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

# def create_mask(image_shape, polygon):
#     mask = np.zeros(image_shape[:2], dtype=np.uint8)
#     cv2.polylines(mask, [polygon], isClosed=True, color=(255, 255, 255), thickness=2)
#     cv2.fillPoly(mask, [polygon], color=(255, 255, 255))
#     return mask

def is_above_line(point, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x, y = point
    offset = 15

    # Calculate the y-coordinate of the line at the given x-coordinate
    line_y = ((y2 - y1) / (x2 - x1)) * (x - x1) + y1
    # print(y, line_y)

    # Compare the y-coordinate of the point to the y-coordinate of the line
    if y + offset < line_y:
        return True
    else:
        return False


def is_left_of_line(point, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x, y = point

    # Calculate the cross product of vectors (line vector) x (point vector)
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    # Check if the cross product is positive (point to the left of the line)
    if cross_product > 0:
        return True
    else:
        return False


def is_right_of_line(point, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x, y = point

    # Calculate the cross product of vectors (line vector) x (point vector)
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    # Check if the cross product is positive (point to the right of the line)
    if cross_product < 0:
        return True
    else:
        return False


video_name = "s2"
json_folder_path = f"output/{video_name}"
config_file_path = f"config/{video_name}.json"
output_video = f"output_video/{video_name}_spatio.avi"
output_frames = f"output_frames/{video_name}_spatio"

# # define detection area
# five_meter = [[965, 610], [965, 715], [1860, 685], [1580, 615]]
# five_meter = np.array(five_meter)
# five_meter = five_meter.reshape((-1, 1, 2))
#
# ten_meter = [[295, 610], [165, 715], [965, 610], [965, 715]]
# ten_meter = np.array(ten_meter)
# ten_meter = ten_meter.reshape((-1, 1, 2))
#
# # create area mask
# five_meter_mask = create_mask(frame.shape, five_meter)
# cv2.imshow("5 meter Mask", five_meter_mask)
# cv2.waitKey(0)

## define lines
# gnd_line = [[1894, 644], [142, 659]]
# five_meter_line = [[960, 330], [960, 805]]
# ten_meter_line = [[195, 330], [195, 805]]
# direction = "left"  # direction towards
with open(config_file_path, 'r') as json_file:
    data = json.load(json_file)
    gnd_line = data['gnd_line']
    five_meter_line = data['five_meter_line']
    ten_meter_line = data['ten_meter_line']
    direction = "left"  # direction towards

## init frame counters
# Performance variables: time to 5m and 10m
start_trigger = False
five_meter_trigger = False
five_meter_counter = 0
ten_meter_counter = 0

# Spatiotemporal variables
start_foot = ""     # back foot
step_start = False
step_length_counter = 0
step_contact_counter = 0
step_flight_counter = 0

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
# get frame dimension
frame_path = "frames/s2/frame_000000000000.jpg"
frame = cv2.imread(frame_path)
height, width, _ = frame.shape
video_writer = cv2.VideoWriter(output_video,
                               cv2.VideoWriter_fourcc(*'XVID'),
                               30,
                               (width, height))

## main loop
for filename in os.listdir(json_folder_path):
    # Check if the path is a file
    if os.path.isfile(os.path.join(json_folder_path, filename)):
        # read JSON file
        numbers = re.findall(r'_(\d+)_', filename)
        frame_number = ''.join(numbers)
        # filled_number = str(frame_number).zfill(12)
        # json_file_path = f"output/{video_name}/{video_name}_{filled_number}_keypoints.json"
        # json_data = read_json_file(json_file_path)
        json_data = read_json_file(os.path.join(json_folder_path, filename))

        # read frame
        frame_path = f"frames/s2/frame_{frame_number}.jpg"
        frame = cv2.imread(frame_path)
        # print(frame.shape)

        try:
            keypoints = json_data['people'][0]['pose_keypoints_2d']

            # JSON read check
            # print(len(keypoints))   # 75 -> 25 keypoints x3, x, y, conf

            # extract keypoints to keypoints_dict for easier interpretation
            keypoints_dict = [{0: ""}]
            for i in range(0, len(keypoints), 3):
                # extract keypoints
                keypoint_num = int(i/3)
                point_to_draw = (round(keypoints[i]), round(keypoints[i+1]))
                # print(keypoint_num, point_to_draw)
                keypoints_dict[0][keypoint_num] = point_to_draw

                # draw on points on frame
                cv2.circle(frame, point_to_draw, 2, (0, 255, 0), 2)
                if keypoint_num == 19 or keypoint_num == 22:    # indicating which toe detected
                    cv2.putText(frame, str(keypoint_num), point_to_draw, cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

            # print(keypoints_dict)

            # detect if foot lifted
            if not start_trigger and not five_meter_trigger:
                if is_above_line(keypoints_dict[0][19], gnd_line):
                    print("LBigToe off ground", frame_number)
                    start_trigger = True
                    start_foot = "left"
                elif is_above_line(keypoints_dict[0][22], gnd_line):
                    print("RBigToe off ground", frame_number)
                    start_trigger = True
                    start_foot = "right"


            # detect if cross line
            if direction == "left" and start_trigger:
                ten_meter_counter += 1
                if not five_meter_trigger:
                    five_meter_counter += 1
                if is_left_of_line(keypoints_dict[0][8], five_meter_line) and not five_meter_trigger:
                    print("five meters", five_meter_counter)
                    five_meter_trigger = True
                elif is_left_of_line(keypoints_dict[0][8], ten_meter_line):
                    print("ten meters", ten_meter_counter)
                    start_trigger = False

            elif direction == "right" and start_trigger:
                ten_meter_counter += 1
                if not five_meter_trigger:
                    five_meter_counter += 1
                if is_right_of_line(keypoints_dict[0][8], five_meter_line) and not five_meter_trigger:
                    print("five meters", five_meter_counter)
                    five_meter_trigger = True
                elif is_right_of_line(keypoints_dict[0][8], ten_meter_line):
                    print("ten meters", ten_meter_counter)
                    start_trigger = False


            if start_trigger:
                if start_foot == "left":
                    if not is_above_line(keypoints_dict[0][19], gnd_line):  # RBigToe
                        step_start = True
                        step_length_counter += 1
                        step_contact_counter += 1
                        print("left", frame_number)
                    elif is_above_line(keypoints_dict[0][22], gnd_line) and step_start:  # both feet above ground
                        step_flight_counter += 1
                        print("flying", frame_number)
                    elif step_start:
                        print("left foot", step_length_counter, step_contact_counter, step_flight_counter)
                        start_foot = "right"
                elif start_foot == "right":
                    if not is_above_line(keypoints_dict[0][22], gnd_line):  # RBigToe
                        step_start = True
                        step_length_counter += 1
                        step_contact_counter += 1
                        print("right", frame_number)
                    elif is_above_line(keypoints_dict[0][19], gnd_line) and step_start:    # both feet above ground
                        step_flight_counter += 1
                        print("flying", frame_number)
                    elif step_start:
                        print("right foot", step_length_counter, step_contact_counter, step_flight_counter)
                        start_foot = "left"




        except IndexError:
            # print("no person detected in this frame", json_data)
            pass

        finally:
            # frame number
            cv2.putText(frame, str(frame_number), (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # # draw area on frame
            # cv2.polylines(frame, [five_meter], isClosed=True, color=(0, 255, 0), thickness=2)

            # draw line on frame
            cv2.line(frame, gnd_line[0], gnd_line[1], (0, 0, 255), 2)
            cv2.line(frame, five_meter_line[0], five_meter_line[1], (0, 0, 255), 2)
            cv2.line(frame, ten_meter_line[0], ten_meter_line[1], (0, 0, 255), 2)
            cv2.imshow('frame', frame)
            # video_writer.write(frame)
            # cv2.imwrite(f"{output_frames}/{frame_number}.jpg", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# cv2.imshow(f"{frame_number}", frame)
# cv2.waitKey(0)
cv2.destroyAllWindows()

