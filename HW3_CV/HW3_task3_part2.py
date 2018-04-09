#!/usr/bin/env python3
"""
@date: 03/20/2018
@authors: Team SoJ (JJ Lim, So Jin Oh)

This is for COMP380: Bodies/Mind AI Robotics Homework 3: Computer Vision Task3 Part2. Works like
the automatic code on the PixyCam does, but runs on the webcam video feed. Given an image, and a
predetermined range of hues to match, finds blobs of the appropriate hues, and reports their bounding
rectangles by drawing them on the image. Runs on a video feed.
"""

import cv2
import numpy as np
import time

def initORB(numFeatures=25):
    """
    Initiate STAR detection (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) (https://stackoverflow.com/questions/32702433/opencv-orb-detector-finds-very-few-keypoints)
    :param numFeatures: the number of features that will be detected and computed using ORB
    :return: initiated orb
    """
    return cv2.ORB_create(nfeatures=numFeatures, scoreType=cv2.ORB_FAST_SCORE)


def getCannyEdge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cannyImg = cv2.Canny(gray, 100, 200)
    return cannyImg

def findROIWithHue(frame, hueLower=0, hueUpper=5):
    """
    Find region of interest with the given hue range and draw a rectangle around it.
    Default hue range finds red objects.
    :param frame:
    :param hueLower:
    :param hueUpper:
    :return:
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Change colorspace from BGR to HSV (hue) Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    # h, s, v = cv2.split(hsv)
    # define range of blue color in HSV (https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html)
    hueLower = np.array([hueLower, 100, 100])
    hueUpper = np.array([hueUpper, 255, 255])

    # Threshold the HSV image to get only hue in the given range
    mask = cv2.inRange(hsv, hueLower, hueUpper)


    # Bitwise-AND mask and original image
    hueImage = cv2.bitwise_and(frame, frame, mask=mask)
    edges = getCannyEdge(hueImage)
    st = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    res = cv2.morphologyEx(src=edges, op=cv2.MORPH_CLOSE, kernel=st)


    _, contours, h = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)  # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # cv2.imshow('frame', frame)
    # cv2.imshow('hueImage', hueImage)


    return frame


if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)

    vidCap = cv2.VideoCapture(0)


    while True:
        start_time = time.time()
        gotOne, frame = vidCap.read()

        if (gotOne):
            frame = findROIWithHue(frame, hueLower=0, hueUpper=5)
            cv2.imshow("VideoCam", frame)

        x = cv2.waitKey(10)  # Waiting may be needed for window updating
        char = chr(x & 0xFF)
        if (char == 'q'):  # esc == '27'
            break


    cv2.destroyAllWindows()
    vidCap.release()


