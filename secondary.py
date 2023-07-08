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


def printSteps(steps, file):
    for i, s in enumerate(steps):
        if file is not None:
            print(i, s, file=file)
        else:
            print(i, s)


def isPlayer(keypoints):
    head = (keypoints[0], keypoints[1])
    RToe = (keypoints[3*22], keypoints[3*22+1])
    distance = math.hypot(head[0] - RToe[0], head[1] - RToe[1])
    # print(distance)
    if distance > 120:
        return True
    return False


def bodyCM(keypoints_dict):
    index = [1, 2, 5, 8, 9, 12]
    trunk_x = [keypoints_dict[i][0] for i in index]
    trunk_y = [keypoints_dict[i][1] for i in index]
    trunk_x.sort()
    trunk_y.sort()
    mid_ind = len(trunk_x) // 2
    if len(trunk_x) % 2 == 1:
        mid_x = trunk_x[mid_ind]
        mid_y = trunk_y[mid_ind]
    else:
        mid_x = (trunk_x[mid_ind - 1] + trunk_x[mid_ind]) // 2
        mid_y = (trunk_y[mid_ind - 1] + trunk_y[mid_ind]) // 2
    print("bodyCM:", mid_x, mid_y)
    return mid_x, mid_y


def dist_CM2Toe(cm_x, toe_x):
    return abs(cm_x - toe_x)


def dist_contact(touchDown_x, toeOff_x):
    return abs(touchDown_x - toeOff_x)


def dist_flight(next_touchDown_x, toeOff_x):
    return abs(next_touchDown_x - toeOff_x)


class stepData():
    start_frame = 0
    end_frame = 0

    step_start_coord = ()
    step_end_coord = ()

    step_contact_counter = 0
    step_flight_counter = 0

    touchDown_dist = 0
    toeOff_dist = 0
    contact_length = 0
    flight_length = 0
    touchDown_cm_coord = ()
    toeOff_cm_coord = ()

    def __init__(self, start_frame):
        self.step_start_coord = ()
        self.step_end_coord = ()
        self.step_contact_counter = 0
        self.step_flight_counter = 0
        self.touchDown_dist = 0
        self.toeOff_dist = 0
        self.contact_length = 0
        self.flight_length = 0
        self.touchDown_cm_coord = ()
        self.toeOff_cm_coord = ()
        self.start_frame = start_frame

    def __str__(self):
        return f'start: frame {self.start_frame} {self.step_start_coord}, ' \
               f'end: frame {self.end_frame} {self.step_end_coord}, ' \
               f'contact frames: {self.step_contact_counter}, ' \
               f'flight frames: {self.step_flight_counter}' \
               f'\ntouch down dist: {self.touchDown_dist}, ' \
               f'toe off dist: {self.touchDown_dist}, ' \
               f'contact length: {self.contact_length}, ' \
               f'flight length: {self.flight_length}'


debug = False
video_name = "s2"   # without extension
json_folder_path = f"output/{video_name}"
config_file_path = f"config/{video_name}.json"
output_video = f"output_video/{video_name}.avi"
output_frames_folder = f"output_frames/{video_name}"

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
toe_off = False
steps = []
step_offset = 7

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

# get frame dimension for video writer
if not debug:
    frame_path = f"frames/{video_name}/frame_000000000000.jpg"
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(output_video,
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   30,
                                   (width, height))

# init keypoints list
keypoints_dict_list = []
buffer_size = 10

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
            keypoints_dict = {"frame": frame_number}
            for i in range(0, len(keypoints), 3):
                # extract keypoints
                keypoint_num = int(i/3)
                point_to_draw = (round(keypoints[i]), round(keypoints[i+1]))
                # print(keypoint_num, point_to_draw)
                keypoints_dict[keypoint_num] = point_to_draw
                # draw on points on frame
                cv2.circle(frame, point_to_draw, 2, (0, 255, 0), 2)
                if keypoint_num == 19 or keypoint_num == 22:    # indicating which toe detected
                    cv2.putText(frame, str(keypoint_num), point_to_draw, cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

            # compare with prev frame
            if len(keypoints_dict_list) > 0:
                LeftBigToe_prev_x = keypoints_dict_list[-1][19][0]
                RightBigToe_prev_x = keypoints_dict_list[-1][22][0]

                # swap the x coord of keypoints if differ by more than the margin
                swap_margin = 40
                # print("before swap", keypoints_dict[19], keypoints_dict[22])
                if keypoints_dict[19][0] > LeftBigToe_prev_x + swap_margin \
                        or keypoints_dict[22][0] > RightBigToe_prev_x + swap_margin \
                        or keypoints_dict[19][0] < LeftBigToe_prev_x - swap_margin \
                        or keypoints_dict[22][0] < RightBigToe_prev_x - swap_margin:
                    tmp = keypoints_dict[19]
                    keypoints_dict[19] = keypoints_dict[22]
                    keypoints_dict[22] = tmp
                    # print("after swap", keypoints_dict[19], keypoints_dict[22])
                    cv2.circle(frame, keypoints_dict[19], 2, (0, 0, 255), 2)
                    cv2.putText(frame, str(19), keypoints_dict[19], cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, keypoints_dict[22], 2, (0, 0, 255), 2)
                    cv2.putText(frame, str(22), keypoints_dict[22], cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)

            # print(keypoints_dict)
            keypoints_dict_list.append(keypoints_dict)
            if len(keypoints_dict_list) > buffer_size:
                keypoints_dict_list.pop(0)


            # detect if foot lifted
            if not start_trigger and not five_meter_trigger:
                # two_toe_distance = math.hypot(keypoints_dict[19][0] - keypoints_dict[22][0], keypoints_dict[19][1] - keypoints_dict[22][1])
                # if two_toe_distance > 60:   # inaccurate keypoint problem eg overlapping toes
                if is_above_line(keypoints_dict[19], gnd_line, perf_offset):
                    print(f"LBigToe off ground {frame_number}")
                    start_frame = frame_number
                    start_trigger = True
                    start_foot = "left"
                elif is_above_line(keypoints_dict[22], gnd_line, perf_offset):
                    print(f"RBigToe off ground {frame_number}")
                    start_frame = frame_number
                    start_trigger = True
                    start_foot = "right"


            # Performance variables: detect if cross line
            if direction == "left" and start_trigger:
                ten_meter_counter += 1
                if not five_meter_trigger:
                    five_meter_counter += 1
                if is_left_of_line(keypoints_dict[8], five_meter_line) and not five_meter_trigger:
                    print(f"five meters {five_meter_counter}")
                    five_meter_trigger = True
                elif is_left_of_line(keypoints_dict[8], ten_meter_line):
                    print(f'ten meters {ten_meter_counter}')
                    start_trigger = False

            elif direction == "right" and start_trigger:
                ten_meter_counter += 1
                if not five_meter_trigger:
                    five_meter_counter += 1
                if is_right_of_line(keypoints_dict[8], five_meter_line) and not five_meter_trigger:
                    print(f"five meters {five_meter_counter}")
                    five_meter_trigger = True
                elif is_right_of_line(keypoints_dict[8], ten_meter_line):
                    print(f'ten meters {ten_meter_counter}')
                    start_trigger = False


            # spatio varaibles
            if start_trigger or step_start:
                if start_foot == "left":
                    if not is_above_line(keypoints_dict[19], gnd_line, step_offset):  # LBigToe on line
                        if not step_start:
                            step_start = True
                            step = stepData(int(frame_number))
                            step.step_contact_counter += 1
                            step.step_start_coord = keypoints_dict[19]
                            print("left", frame_number)

                            # touch down distance
                            step.touchDown_cm_coord = bodyCM(keypoints_dict)
                            cv2.circle(frame, step.touchDown_cm_coord, 2, (0, 0, 255), 2)
                            cv2.putText(frame, "CM", step.touchDown_cm_coord, cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2, cv2.LINE_AA)
                            step.touchDown_dist = dist_CM2Toe(step.touchDown_cm_coord[0], keypoints_dict[19][0])
                        else:
                            # if is_above_line(keypoints_dict[22], gnd_line, step_offset):
                            step.step_contact_counter += 1
                            # print("left contact", frame_number)
                    elif is_above_line(keypoints_dict[22], gnd_line, step_offset) and step_start:
                        if is_above_line(keypoints_dict[19], gnd_line, step_offset):  # both feet above gnd
                            step.step_flight_counter += 1
                            toe_off = True
                            # print("left flight", frame_number)

                            # toe off distance
                            if step.toeOff_dist == 0:
                                print("toe off")
                                step.toeOff_cm_coord = bodyCM(keypoints_dict)
                                cv2.circle(frame, step.toeOff_cm_coord, 2, (0, 0, 255), 2)
                                cv2.putText(frame, "CM", step.toeOff_cm_coord, cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 0, 255), 2, cv2.LINE_AA)
                                step.toeOff_dist = dist_CM2Toe(step.toeOff_cm_coord[0], keypoints_dict[19][0])
                                step.contact_length = dist_contact(step.touchDown_cm_coord[0], step.toeOff_cm_coord[0])
                    elif step_start and toe_off:    # enf of step
                        print("end step")
                        step.step_end_coord = keypoints_dict[22]
                        end_cm = bodyCM(keypoints_dict)
                        cv2.circle(frame, end_cm, 2, (0, 0, 255), 2)
                        cv2.putText(frame, "CM", end_cm, cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                        step.flight_length = dist_flight(end_cm[0], step.toeOff_cm_coord[0])
                        step.end_frame = int(frame_number)
                        print(f'{str(step)}, frame {int(frame_number)}')
                        steps.append(step)
                        step_start = False
                        start_foot = "right"
                        # printSteps(steps)

                elif start_foot == "right":
                    if not is_above_line(keypoints_dict[22], gnd_line, step_offset):      # RBigToe on line
                        if not step_start:
                            step_start = True
                            step = stepData(int(frame_number))
                            step.step_contact_counter += 1
                            step.step_start_coord = keypoints_dict[22]
                            print("right", frame_number)

                            # touch down distance
                            step.touchDown_cm_coord = bodyCM(keypoints_dict)
                            cv2.circle(frame, step.touchDown_cm_coord, 2, (0, 0, 255), 2)
                            cv2.putText(frame, "CM", step.touchDown_cm_coord, cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2, cv2.LINE_AA)
                            step.touchDown_dist = dist_CM2Toe(step.touchDown_cm_coord[0], keypoints_dict[22][0])
                        else:
                            # if is_above_line(keypoints_dict[19], gnd_line, step_offset):
                            step.step_contact_counter += 1
                            # print("right contact", frame_number)
                    elif is_above_line(keypoints_dict[19], gnd_line, step_offset) and step_start:
                        if is_above_line(keypoints_dict[22], gnd_line, step_offset):    # both feet above gnd
                            step.step_flight_counter += 1
                            toe_off = True
                            # print("right flight", frame_number)

                            # toe off distance
                            if step.toeOff_dist == 0:
                                print(f"toe off {frame_number}")
                                step.toeOff_cm_coord = bodyCM(keypoints_dict)
                                cv2.circle(frame, step.toeOff_cm_coord, 2, (0, 0, 255), 2)
                                cv2.putText(frame, "CM", step.toeOff_cm_coord, cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 0, 255), 2, cv2.LINE_AA)
                                step.toeOff_dist = dist_CM2Toe(step.toeOff_cm_coord[0], keypoints_dict[22][0])
                                step.contact_length = dist_contact(step.touchDown_cm_coord[0], step.toeOff_cm_coord[0])
                    elif step_start and toe_off:
                        print(f"end step right {frame_number}")
                        step.step_end_coord = keypoints_dict[19]
                        end_cm = bodyCM(keypoints_dict)
                        cv2.circle(frame, end_cm, 2, (0, 0, 255), 2)
                        cv2.putText(frame, "CM", end_cm, cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                        step.flight_length = dist_flight(end_cm[0], step.toeOff_cm_coord[0])
                        step.end_frame = int(frame_number)
                        print(f'{str(step)}, frame {int(frame_number)}')
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
            if not debug:
                video_writer.write(frame)
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
print(f"pref_offset: {perf_offset}",  file=log_file)
print(f"step_offset: {step_offset}", file=log_file)
print(f"start frame {int(start_frame)}", file=log_file)
print(f"five meters {five_meter_counter}", file=log_file)
print(f"ten meters {ten_meter_counter}", file=log_file)

# step calculations
print("\nsteps debug")
printSteps(steps, log_file)
printSteps(steps, None)

log_file.close()
cv2.destroyAllWindows()