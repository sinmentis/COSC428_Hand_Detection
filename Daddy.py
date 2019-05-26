import cv2
import numpy as np
import copy
import math
from sklearn.metrics import pairwise

# Parameters Defines
IOR_X = 0.5  # start point/total width
IOR_Y = 0.8    # start point/total width
threshold = 20          # BINARY threshold
blurValue = 11          # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
BLUE = (255, 0, 0)
bgModel = None

# Flag Variables
triggerSwitch = False   # if true, keyborad simulator works
isBgCaptured = False


def count(thresholded, segmented):
    """To count the number of fingers in the segmented hand region"""
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = (extreme_left[0] + extreme_right[0]) / 2
    cY = (extreme_top[1] + extreme_bottom[1]) / 2

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count


def printThreshold(thr):
    """Print out Threshold"""
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    global bgModel
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
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
            return True, cnt
    return False, 0


def read_key():
    global isBgCaptured, triggerSwitch, bgModel
    k = cv2.waitKey(10)
    result = ""
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        result = "break"
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('!!!Background Captured!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print('!!!Trigger On!!!')

    return result


def image_process(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (blurValue, blurValue), 0)
    frame_fil = cv2.bilateralFilter(frame_blur, 5, 50, 100)  # Smoothing Filter
    return frame_fil


if __name__ == "__main__":

    # Camera
    camera = cv2.VideoCapture(-1)
    camera.set(10, 200)            # Brightness
    _, frame = camera.read()
    [F_Y, F_X] = [frame.shape[0], frame.shape[1]]                               # Load Resolution

    while camera.isOpened():
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)        # Cancel Mirror
        cv2.rectangle(frame, pt1=(int(IOR_X * F_X), 0),
                      pt2=(frame.shape[1], int(IOR_Y * F_Y)),
                      color=BLUE, thickness=2)  # IOR
        cv2.imshow('Original', frame)
        frame_process = image_process()

        if isBgCaptured == 1:
            frame_IOR = frame_process[0:int(IOR_Y * F_Y),
                        int(IOR_X * F_X):F_X]  # clip the ROI
            frame_IOR_removeBG = removeBG(frame_IOR)
            # convert the image into binary image
            _, frame_IOR_thre = cv2.threshold(frame_IOR_removeBG, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('removeBG threshold', frame_IOR_thre)


            # get the coutours
            thresh1 = copy.deepcopy(frame_IOR_thre)
            _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

                isFinishCal, cnt = calculateFingers(res,drawing)
                if triggerSwitch is True:
                    if isFinishCal is True and cnt <= 2:
                        print(cnt)


            cv2.imshow('output', drawing)

        # Keyboard OP
        read_key()
