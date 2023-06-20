import os
import json

import cv2

mouse_pts = []
line_pts = []
counter = 0

def mousePoint(event,x,y,flags,params):
    global counter, line_counter, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pts.append([x, y])
        print(f"line{len(line_pts)+1}", x, y)
        counter += 1
    if counter == 2:
        line_pts.append(mouse_pts)
        cv2.line(frame, mouse_pts[0], mouse_pts[1], (0, 0, 255), 2)
        counter = 0
        mouse_pts = []
    if len(line_pts) == 3:
        cv2.setMouseCallback("frame", lambda *args: None)
        cv2.putText(frame, end_txt, (0, 30 * len(instruction_text) + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

video_name = "s2"
frame_folder_path = f"frames/{video_name}"
filename = "frame_000000000000.jpg"
file_path = os.path.join(frame_folder_path, filename)
json_file_path = f"config/{video_name}.json"

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("frame", mousePoint)
instruction_text = ["Press the arrow key of the direction the player is running",
                    "Click two points to form a the line. ",
                    "Click the lines in the following order: ",
                    "1. Ground line - make sure it touches the player's toes",
                    "2. 5 metre line ",
                    "3. 10 metre line "
                    ]
end_txt = "lines recorded in file, press q to exit"


# Check if the path is a file
if os.path.isfile(file_path):
    # only read first frame
    frame = cv2.imread(file_path)
    for i, line in enumerate(instruction_text):
        cv2.putText(frame, str(line), (0,30*i+30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    data = {}

    while True:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 2424832:
            data['direction'] = 'left'
            print(data['direction'])
            cv2.putText(frame, str(data['direction']), (0, 30 * len(instruction_text) + 30*2), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        elif cv2.waitKey(1) == 2555904:
            data['direction'] = 'right'
            print(data['direction'])
            cv2.putText(frame, str(data['direction']), (0, 30 * len(instruction_text) + 30*2), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if len(line_pts) > 2:
        data['gnd_line'] = line_pts[0]
        data['five_meter_line'] = line_pts[1]
        data['ten_meter_line'] = line_pts[2]
        print(data)
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)
    else:
        print("incomplete data")

cv2.destroyAllWindows()
