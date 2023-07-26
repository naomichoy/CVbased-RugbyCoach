from ultralytics import YOLO
import numpy as np
import torch
import cv2
import time

img = './test_images/frame_000000002491.jpg'
vid = "./test_images/P1.mp4"

# Load a model
model = YOLO('yolov8m-seg.pt')  # pretrained YOLOv8n model

cap = cv2.VideoCapture(vid)
# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
frame_number = 0
start_time = time.time()
while cap.isOpened():
    _, frame = cap.read()
    if not _ :
        print("read frame fail")
        break


    results = model(frame, device=0, classes=[32], show_conf=True)
    frame = results[0].plot()

    cv2.putText(frame, str(frame_number), (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    frame_number += 1
    if frame_number == 500:
        print(time.time()-start_time)
        cv2.waitKey(0)

    # if len(results[0].boxes) == 0:
    #     cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# # Run batched inference on a list of images
# results = model(vid, show=True, save=False, stream=True, device=0, classes=[32], show_conf=True)
#
# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Class probabilities for classification outputs
#
#     cv2.waitKey(0)
#     # check class_id, sports ball is 32
#     class_id = np.array(boxes.cls.cpu(), dtype="int")
#     print(class_id)


