import os
import sys
import logging
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


def isPlayer(keypoints):
    head = (keypoints[0], keypoints[1])
    RToe = (keypoints[3 * 22], keypoints[3 * 22 + 1])
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
    # print("bodyCM:", mid_x, mid_y)
    logging.info(f"bodyCM: {mid_x}, {mid_y}")
    return mid_x, mid_y


def drawTrajectory(frame, ball_c_list):
    if len(ball_c_list) > 0:
        for pt in ball_c_list:
            cv2.circle(frame, pt, 2, (32, 165, 218), 2)


def calculate_bottom(bbox_xywh):
    x, y, w, h = bbox_xywh
    bx = x + (w / 2)
    by = y + (h / 2)
    return bx, by


def euclidean_distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    cos_theta = dot_product / (norm_vector1 * norm_vector2)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def vector_between_points(point1, point2):
    return np.array(point2) - np.array(point1)


video_name = "P3_test"  # without extension
fps = 500
true_dist = 0.38   # P1: 0.45m  P2: 0.4m  P3: 0.38m
dist_ratio = 1

save_frames = False
time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
json_folder_path = f"output/{video_name}"
config_file_path = f"config/{video_name}.json"
output_video = f"output_video/{video_name}-{time_now}.avi"
output_frames_folder = f"output_frames/{video_name}-{time_now}"

if not os.path.exists(output_frames_folder) and save_frames:
    os.makedirs(output_frames_folder)

# logfile
log_file = open(f'logs/{video_name}_{time_now}.txt', 'w')
logg_file = f"logs/{video_name}_{time_now}.log"
targets = logging.StreamHandler(sys.stdout), logging.FileHandler(logg_file)
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)


cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

# get frame dimension for video writer
if save_frames:
    frame_folder_path = f"frames/{video_name}"
    folder_walk = os.walk(frame_folder_path)
    frame_name = next(folder_walk)[2][0]
    frame_path = os.path.join(frame_folder_path, frame_name)
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(output_video,
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   30,
                                   (width, height))

# init keypoints list
keypoints_dict_list = []        # dictionary item type: tuple
buffer_size = 5
ball_c_list = []

# kicking control params
touch_thres = 10
ball_leave = False
ball_drop = False

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
            logging.info(f"person in frame {frame_number}: {len(json_data['people'])}")

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
                keypoint_num = int(i / 3)
                point_to_draw = (round(keypoints[i]), round(keypoints[i + 1]))
                # print(keypoint_num, point_to_draw)
                keypoints_dict[keypoint_num] = point_to_draw
                # draw on points on frame
                cv2.circle(frame, point_to_draw, 2, (0, 255, 0), 2)
                if keypoint_num == 19 or keypoint_num == 22:  # indicating which toe detected
                    cv2.putText(frame, str(keypoint_num), point_to_draw, cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

            # compare with prev frames to swap
            if len(keypoints_dict_list) > 0:
                LeftBigToe_prev_x = keypoints_dict_list[-1][19][0]
                RightBigToe_prev_x = keypoints_dict_list[-1][22][0]

                # calculate the mean and median of both toe keypoint
                LeftBigToe_x_list = [item[19][0] for item in keypoints_dict_list]
                RightBigToe_x_list = [item[22][0] for item in keypoints_dict_list]

                LeftBigToe_x_avg = sum(LeftBigToe_x_list) / len(LeftBigToe_x_list)
                RightBigToe_x_avg = sum(RightBigToe_x_list) / len(RightBigToe_x_list)

                LeftBigToe_x_list.sort()
                RightBigToe_x_list.sort()

                mid_ind = len(LeftBigToe_x_list) // 2
                if len(LeftBigToe_x_list) % 2 == 1:
                    LeftBigToe_mid_x = LeftBigToe_x_list[mid_ind]
                    RightBigToe_mid_x = RightBigToe_x_list[mid_ind]
                else:
                    LeftBigToe_mid_x = (LeftBigToe_x_list[mid_ind - 1] + LeftBigToe_x_list[mid_ind]) // 2
                    RightBigToe_mid_x = (RightBigToe_x_list[mid_ind - 1] + RightBigToe_x_list[mid_ind]) // 2

                # swap the x coord of keypoints if differ by more than the margin
                swap_margin = 40
                mid_swap_margin = 60
                # print(frame_number, "before swap", keypoints_dict[19], keypoints_dict[22])
                logging.info(f"{frame_number} before swap {keypoints_dict[19]} {keypoints_dict[22]}")
                prev_bool = keypoints_dict[19][0] > LeftBigToe_prev_x + swap_margin \
                            or keypoints_dict[22][0] > RightBigToe_prev_x + swap_margin \
                            or keypoints_dict[19][0] < LeftBigToe_prev_x - swap_margin \
                            or keypoints_dict[22][0] < RightBigToe_prev_x - swap_margin \
                            or LeftBigToe_prev_x - keypoints_dict[19][0] > LeftBigToe_prev_x - keypoints_dict[22][0]

                # avg_bool = keypoints_dict[19][0] > LeftBigToe_x_avg + swap_margin \
                #             or keypoints_dict[22][0] > RightBigToe_x_avg + swap_margin \
                #             or keypoints_dict[19][0] < LeftBigToe_x_avg - swap_margin \
                #             or keypoints_dict[22][0] < RightBigToe_x_avg - swap_margin

                mid_bool = keypoints_dict[19][0] > LeftBigToe_mid_x + mid_swap_margin \
                           or keypoints_dict[22][0] > RightBigToe_mid_x + mid_swap_margin \
                           or keypoints_dict[19][0] < LeftBigToe_mid_x - mid_swap_margin \
                           or keypoints_dict[22][0] < RightBigToe_mid_x - mid_swap_margin

                if mid_bool and prev_bool:
                    # if LeftBigToe_prev_x - keypoints_dict[19][0] > LeftBigToe_prev_x - keypoints_dict[22][0]:     # diff between prev x of keypoint 19 and both current 19 and 22
                    tmp = keypoints_dict[19]
                    keypoints_dict[19] = keypoints_dict[22]
                    keypoints_dict[22] = tmp
                    # print("after swap", keypoints_dict[19], keypoints_dict[22])
                    logging.info(f"after swap {keypoints_dict[19]} {keypoints_dict[22]}")
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

            # draw CM
            cmpt = bodyCM(keypoints_dict)
            cv2.circle(frame, cmpt, 2, (255, 255, 255), 2)
            cv2.putText(frame, "CM", cmpt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # read ball mask
            mask_json = f'{frame_number}.json'
            mask_path = os.path.join("output_mask_yolo", video_name, mask_json)
            mask_data = read_json_file(mask_path)
            # print(mask_data)
            mask_xy = mask_data['mask']
            bbox_xywh = mask_data['bbox']
            conf = mask_data['conf']

            if len(bbox_xywh) > 0:
                ball_c_list.append((round(bbox_xywh[0]),round(bbox_xywh[1])))
                # print(ball_c_list)
                bx, by = calculate_bottom(bbox_xywh)
            else:   # get median coordinate of pass frames
                bx, by, = 500,  500  # test arbitrary

            # determine kicking foot and ball dropped
            # detect when free fall happens
            # filter free fall detection noise, lower than both knee point
            RKnee3 = keypoints_dict[10][1] + (keypoints_dict[11][1] - keypoints_dict[10][1]) / 4
            LKnee3 = keypoints_dict[13][1] + (keypoints_dict[14][1] - keypoints_dict[13][1]) / 4
            knee_thres = max(RKnee3, LKnee3)
            cv2.circle(frame, (cmpt[0], int(knee_thres)), 2, (255, 255, 255), 2)
            if bbox_xywh[1] > knee_thres:
                if len(ball_c_list) > 1:
                    dy = ball_c_list[-1][1] - ball_c_list[-2][1]    # TODO: * -1 ?
                    dx = ball_c_list[-1][0] - ball_c_list[-2][0]
                    if dx == 0:
                        dydx = 0
                    else:
                        dydx = dy / dx
                        logging.info(f'dydx: {dydx}')
                    if dydx < 0:
                        logging.info("-ve gradient")
                        ball_drop = True
            kicking_foot = "right"

            # determine when ball_leave
            if ball_drop:
                if kicking_foot == "right" and keypoints_dict[22][1] - by > touch_thres:
                    ball_leave = True
                if kicking_foot == "left" and keypoints_dict[19][1] - by > touch_thres:
                    pass

            if ball_leave:
                ball_velocity_sum = 0
                for i in range(-1, -6, -1):
                    # print(i)
                    m = (ball_c_list[i][1] - ball_c_list[i-1][1]) / (ball_c_list[i][0] - ball_c_list[i-1][0]) * -1
                    ball_displacement = euclidean_distance(ball_c_list[i], ball_c_list[i-1]) * m
                    ball_velocity = ball_displacement * dist_ratio / (1/fps)  # wrong formula?? need to multiply by ratio
                    ball_velocity_sum += ball_velocity
                ball_velocity_avg = ball_velocity_sum / 5
                logging.info(f"ball release velocity: {ball_velocity_avg}")

            #  ball on contact. foot speed
            if ball_drop and not ball_leave:
                foot_speed_sum = 0
                thigh_angle_velocity_sum = 0
                knee_angle_velocity_sum = 0
                for i in range(-1, -(len(keypoints_dict_list) - 1), -1):
                    if kicking_foot == "right":
                        # use right ankle point
                        foot_displacement = euclidean_distance(keypoints_dict_list[i][11], keypoints_dict_list[i-1][11])
                        foot_speed = foot_displacement * dist_ratio / (1/fps)
                        foot_speed_sum += foot_speed

                        # calculate direction vector from position vectors (keypoints_dict items are positional vectors)
                        # thigh angles
                        # t_prev_9_h = vector_between_points(keypoints_dict_list[i-1][9], [1,0])  # change this to horizontal??
                        t_prev_9_10 = vector_between_points(keypoints_dict_list[i-1][9], keypoints_dict_list[i-1][10])
                        t_prev_angle = angle_between_vectors([1,0], t_prev_9_10)

                        # t_cur_9_h = vector_between_points(keypoints_dict_list[i][9], [1,0])
                        t_cur_9_10 = vector_between_points(keypoints_dict_list[i][9], keypoints_dict_list[i][10])
                        t_current_angle = angle_between_vectors([1,0], t_cur_9_10)

                        t_angular_velocity = (t_current_angle - t_prev_angle) / (1/fps)   # check sign
                        thigh_angle_velocity_sum += t_angular_velocity

                        # knee angles
                        # k_prev_10_h = vector_between_points(keypoints_dict_list[i-1][10], [1,0])  # TODO: change this to horizontal
                        k_prev_10_11 = vector_between_points(keypoints_dict_list[i-1][10], keypoints_dict_list[i-1][11])
                        k_prev_angle = angle_between_vectors([1,0], k_prev_10_11)

                        # k_cur_10_h = vector_between_points(keypoints_dict_list[i][10], [1,0])
                        k_cur_10_11 = vector_between_points(keypoints_dict_list[i][10], keypoints_dict_list[i][11])
                        k_current_angle = angle_between_vectors([1,0], k_cur_10_11)

                        k_angular_velocity = (k_current_angle - k_prev_angle) / (1 / fps)  # check sign
                        knee_angle_velocity_sum += k_angular_velocity

                        # cv2.waitKey(0)

                foot_speed_avg = foot_speed_sum / 5
                thigh_angle_velocity_avg = thigh_angle_velocity_sum / 5
                knee_angle_velocity_avg = knee_angle_velocity_sum / 5
                logging.info(f"foot velocity: {foot_speed_avg}")
                logging.info(f"thigh angular velocity: {thigh_angle_velocity_avg}")
                logging.info(f"knee angular velocity: {knee_angle_velocity_avg}")

        except IndexError:
            # print("no person detected in this frame", json_data)
            pass

        finally:
            # frame number
            cv2.putText(frame, str(frame_number), (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            drawTrajectory(frame, ball_c_list)

            cv2.imshow('frame', frame)
            if save_frames:
                video_writer.write(frame)
                cv2.imwrite(f"{output_frames_folder}/{frame_number}.jpg", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

                