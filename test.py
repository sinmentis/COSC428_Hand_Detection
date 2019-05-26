import cv2
import numpy as np
import copy
import math
from sklearn.metrics import pairwise

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8    # start point/total width
threshold = 60          #  BINARY threshold
blurValue = 41          # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
BLUE = (255, 0, 0)

# variables
isBgCaptured = 0        # bool, whether the background captured
triggerSwitch = False   # if true, keyborad simulator works


camera = cv2.VideoCapture(-1)
camera.set(10, 200)  # Brightness

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), color=BLUE, thickness=2)
    cv2.imshow('original', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
