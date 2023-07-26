from ultralytics import YOLO
import numpy as np
import torch
import cv2

img = './test_images/frame_000000002491.jpg'
vid = "./test_images/P1.mp4"

# Load a model
model = YOLO('yolov8m-seg.pt')  # pretrained YOLOv8n model

cap = cv2.VideoCapture(vid)
# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
while cap.isOpened():
    _, frame = cap.read()
    if not _ :
        print("read frame fail")
        break

    results = model(frame, device=0, classes=[32], show_conf=True)
    frame = results[0].plot()
    cv2.imshow('frame', frame)

    if len(results[0].boxes) == 0:
        cv2.waitKey(0)

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


