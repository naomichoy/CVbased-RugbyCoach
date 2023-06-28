import os
import sys
import re
import json
import cv2
from PIL import Image
import numpy as np
import time
import math


def read_json_file(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

# def create_mask(image_shape, polygon):
#     mask = np.zeros(image_shape[:2], dtype=np.uint8)
#     cv2.polylines(mask, [polygon], isClosed=True, color=(255, 255, 255), thickness=2)
#     cv2.fillPoly(mask, [polygon], color=(255, 255, 255))
#     return mask

def is_above_line(point, line, offset):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x, y = point
    # offset = 15

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

def printSteps(steps):
    for i, s in enumerate(steps):
        print(i, s)

def isPlayer(keypoints):
    head = (keypoints[0], keypoints[1])
    RToe = (keypoints[3*22], keypoints[3*22+1])
    distance = math.hypot(head[0] - RToe[0], head[1] - RToe[1])
    # print(distance)
    if distance > 120:
        return True
    return False


class stepData():
    step_start_coord = ()
    step_end_coord = ()
    step_contact_counter = 0
    step_flight_counter = 0

    def __init__(self):
        self.step_start_coord = ()
        self.step_end_coord = ()
        self.step_contact_counter = 0
        self.step_flight_counter = 0

    def __str__(self):
        return f'start: {self.step_start_coord}, ' \
                f'end: {self.step_end_coord},' \
               f'contact: {self.step_contact_counter}, ' \
               f'flight: {self.step_flight_counter}'


video_name = "s4"   # without extension
json_folder_path = f"output/{video_name}"
config_file_path = f"config/{video_name}.json"
output_video = f"output_video/{video_name}_spatio.avi"
output_frames_folder = f"output_frames/{video_name}_spatio"

if not os.path.exists(output_frames_folder):
    os.makedirs(output_frames_folder)

# logfile
log_file = open(f'logs/{video_name}_{time.strftime("%Y%m%d-%H%M%S", time.localtime())}.txt', 'w')

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
    direction = data['direction']  # direction towards

## init frame counters
# Performance variables: time to 5m and 10m
start_frame = 0
start_trigger = False
five_meter_trigger = False
five_meter_counter = 0
ten_meter_counter = 0
perf_offset = 15

# Spatiotemporal variables
start_foot = ""     # back foot
step_start = False
steps = []
step_offset = 7

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

# get frame dimension
frame_path = f"frames/{video_name}/frame_000000000000.jpg"
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
        frame_path = f"frames/{video_name}/frame_{frame_number}.jpg"
        frame = cv2.imread(frame_path)
        # print(frame.shape)

        try:
            keypoints = json_data['people'][0]['pose_keypoints_2d']
            # number of people check
            # print(f"person in frame {frame_number}: {len(json_data['people'])}")

            for p in json_data['people']:
                next_keypoints = p['pose_keypoints_2d']
                if not isPlayer(next_keypoints):
                    if len(json_data['people']) < 2:
                        # print("not player!")
                        raise IndexError
                else:
                    keypoints = next_keypoints
                    break

            # JSON read keypoint check
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
                two_toe_distance = math.hypot(keypoints_dict[0][19][0] - keypoints_dict[0][22][0], keypoints_dict[0][19][1] - keypoints_dict[0][22][1])
                if two_toe_distance > 60:   # inaccurate keypoint problem eg overlapping toes
                    if is_above_line(keypoints_dict[0][19], gnd_line, perf_offset):
                        print(f"LBigToe off ground {frame_number}")
                        start_frame = frame_number
                        start_trigger = True
                        start_foot = "left"
                    elif is_above_line(keypoints_dict[0][22], gnd_line, perf_offset):
                        print(f"RBigToe off ground {frame_number}")
                        start_frame = frame_number
                        start_trigger = True
                        start_foot = "right"


            # Performance variables: detect if cross line
            if direction == "left" and start_trigger:
                ten_meter_counter += 1
                if not five_meter_trigger:
                    five_meter_counter += 1
                if is_left_of_line(keypoints_dict[0][8], five_meter_line) and not five_meter_trigger:
                    print(f"five meters {five_meter_counter}")
                    five_meter_trigger = True
                elif is_left_of_line(keypoints_dict[0][8], ten_meter_line):
                    print(f'ten meters {ten_meter_counter}')
                    start_trigger = False

            elif direction == "right" and start_trigger:
                ten_meter_counter += 1
                if not five_meter_trigger:
                    five_meter_counter += 1
                if is_right_of_line(keypoints_dict[0][8], five_meter_line) and not five_meter_trigger:
                    print(f"five meters {five_meter_counter}")
                    five_meter_trigger = True
                elif is_right_of_line(keypoints_dict[0][8], ten_meter_line):
                    print(f'ten meters {ten_meter_counter}')
                    start_trigger = False


            # spatio varaibles
            if start_trigger or step_start:
                if start_foot == "left":
                    if not is_above_line(keypoints_dict[0][19], gnd_line, step_offset):  # LBigToe on line
                        if not step_start:
                            step_start = True
                            step = stepData()
                            step.step_contact_counter += 1
                            step.step_start_coord = keypoints_dict[0][19]
                            print("left", frame_number, file=log_file)
                        else:
                            # if is_above_line(keypoints_dict[0][22], gnd_line, step_offset):
                            step.step_contact_counter += 1
                            print("left contact", frame_number, file=log_file)
                    elif is_above_line(keypoints_dict[0][22], gnd_line, step_offset) and step_start:
                        if is_above_line(keypoints_dict[0][19], gnd_line, step_offset):  # both feet above gnd
                            step.step_flight_counter += 1
                            print("left flight", frame_number, file=log_file)
                    elif step_start:
                        step.step_end_coord = keypoints_dict[0][22]
                        print(str(step), file=log_file)
                        steps.append(step)
                        step_start = False
                        start_foot = "right"
                        # printSteps(steps)

                elif start_foot == "right":
                    if not is_above_line(keypoints_dict[0][22], gnd_line, step_offset):      # RBigToe on line
                        if not step_start:
                            step_start = True
                            step = stepData()
                            step.step_contact_counter += 1
                            step.step_start_coord = keypoints_dict[0][22]
                            print("right", frame_number, file=log_file)
                        else:
                            # if is_above_line(keypoints_dict[0][19], gnd_line, step_offset):
                            step.step_contact_counter += 1
                            print("right contact", frame_number, file=log_file)
                    elif is_above_line(keypoints_dict[0][19], gnd_line, step_offset) and step_start:
                        if is_above_line(keypoints_dict[0][22], gnd_line, step_offset):    # both feet above gnd
                            step.step_flight_counter += 1
                            print("right flight", frame_number, file=log_file)
                    elif step_start:
                        step.step_end_coord = keypoints_dict[0][19]
                        print(str(step), file=log_file)
                        steps.append(step)
                        step_start = False
                        start_foot = "left"
                        # printSteps(steps)


        except IndexError:
            # print("no person detected in this frame", json_data)
            pass

        finally:
            # frame number
            cv2.putText(frame, str(frame_number), (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"start frame {int(start_frame)}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frames to five meters {five_meter_counter}", (0, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frames to ten meters {ten_meter_counter}", (0, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # # draw area on frame
            # cv2.polylines(frame, [five_meter], isClosed=True, color=(0, 255, 0), thickness=2)

            # draw line on frame
            cv2.line(frame, gnd_line[0], gnd_line[1], (0, 0, 255), 2)
            cv2.line(frame, five_meter_line[0], five_meter_line[1], (0, 0, 255), 2)
            cv2.line(frame, ten_meter_line[0], ten_meter_line[1], (0, 0, 255), 2)
            cv2.imshow('frame', frame)
            # video_writer.write(frame)
            cv2.imwrite(f"{output_frames_folder}/{frame_number}.jpg", frame)

            # for debug
            # if int(frame_number) > 600 and int(frame_number) < 960:
            #     cv2.waitKey(0)
            # elif cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# cv2.imshow(f"{frame_number}", frame)
# cv2.waitKey(0)

print('\nSummary:', file=log_file)
print(f"start frame {int(start_frame)}", file=log_file)
print(f"five meters {five_meter_counter}", file=log_file)
print(f"ten meters {ten_meter_counter}", file=log_file)

# step calculations
print("\nsteps debug", file=log_file)
printSteps(steps)

log_file.close()
cv2.destroyAllWindows()