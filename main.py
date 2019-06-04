import cv2
import numpy as np
import copy
import math


# Parameters Defines
IOR_X = 0.5  # start point/total width
IOR_Y = 0.8    # start point/total width
BLUE_THRESHOLD = 11          # GaussianBlur parameter
BACKGROUND_THRESHOLD = 50
THRESHOLD = 60
LEARNING_RATE = 0
BLUE = (255, 0, 0)
bgModel = None

# Flag Variables
isBgCaptured = False


def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=LEARNING_RATE)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    img = cv2.bitwise_and(frame, frame, mask=fgmask)
    return img


def calculateFingers(res, drawing):
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            print("Fingers", cnt)


def read_key():
    global isBgCaptured, bgModel
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, BACKGROUND_THRESHOLD)
        isBgCaptured = 1
        print('!!!Background Captured!!!')


def get_frame(camera):
    """Read from camera, draw the ROI"""
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)  # Cancel Mirror
    cv2.rectangle(frame, pt1=(int(IOR_X * F_X), 0),
                  pt2=(frame.shape[1], int(IOR_Y * F_Y)),
                  color=BLUE, thickness=2)  # IOR
    cv2.imshow('Original', frame)
    return frame


def frame_process(frame_BG):
    frame_IOR = frame_BG[0:int(IOR_Y * F_Y), int(IOR_X * F_X):F_X]  # Find ROI
    frame_gray = cv2.cvtColor(frame_IOR, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (BLUE_THRESHOLD, BLUE_THRESHOLD), 0)
    frame_fil = cv2.bilateralFilter(frame_blur, 5, 50, 100)  # Smoothing Filter
    ret, thresh = cv2.threshold(frame_fil, THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imshow('ori', thresh)
    return thresh, frame_IOR

if __name__ == "__main__":
    camera = cv2.VideoCapture(-1)
    camera.set(10, 200)            # Brightness
    _, frame = camera.read()
    [F_Y, F_X] = [frame.shape[0], frame.shape[1]]                                # Load Resolution

    while camera.isOpened():
        frame = get_frame(camera)                                                # Init and Read frame
        if isBgCaptured == 1:
            frame_BG = removeBG(frame)                                 # Remove Background
            frame_thresh, frame_IOR = frame_process(frame_BG)

            # get the coutours
            thresh1 = copy.deepcopy(frame_thresh)
            _, contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i
                res = contours[ci]
                hull = cv2.convexHull(res)
                drawing = np.zeros(frame_IOR.shape, np.uint8)
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
                calculateFingers(res, drawing)
            cv2.imshow('output', drawing)

        read_key()