#!/usr/bin/env python3
"""
@class: COMP380 AI Robotics
@date: 03/20/2018
@authors: Team SoJ (JJ Lim, So Jin Oh)

This is for Homework 3: Computer Vision
"""

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import OrderedDict
from PIL import Image
# import pytesseract  #Python Tesseract; Installation required.
import os

def initORB(numFeatures=25):
    """
    Initiate STAR detection (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) (https://stackoverflow.com/questions/32702433/opencv-orb-detector-finds-very-few-keypoints)
    :param numFeatures: the number of features that will be detected and computed using ORB
    :return: initiated orb
    """
    return cv2.ORB_create(nfeatures=numFeatures, scoreType=cv2.ORB_FAST_SCORE)

def showORB(img, imgKeypoints):
    # Initiate STAR detection (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) (https://stackoverflow.com/questions/32702433/opencv-orb-detector-finds-very-few-keypoints)
    orb = initORB()

    img2 = cv2.drawKeypoints(img, imgKeypoints, None, color=(0, 255, 0))
    cv2.imshow("ORB KeyPoints", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def computeORB(img):
    """
    Detect keypoints and desriptors of an image using ORB.
    :param img: a processed image with cv2.imread(imgFilePath)
    :return: keypoints of the image, descriptors of the image
    """
    orb = initORB()

    # keypoints = orb.detect(image=img)
    keypoints, des = orb.detectAndCompute(img, None)

    return keypoints, des

def readLetterFeatures(dir):
    imgFiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    letterFeatureDict = {}
    for imgFile in imgFiles:
        if (imgFile.endswith('.png')):
            img = cv2.imread(dir+"/" + imgFile)
            kp, des = computeORB(img)
            imgName = imgFile.strip(".png")
            letterFeatureDict.update({imgName: (kp, des)})
    return letterFeatureDict



def getNumMatchedDict(targetImgsDir, queryImg):
    # queryImg = cv2.imread(queryImg)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)

    flanner = cv2.FlannBasedMatcher(index_params, search_params)

    letterFeatureDict = readLetterFeatures(targetImgsDir)

    kpQuery, desQuery = computeORB(queryImg)

    matchedKPs = {}
    for letter in letterFeatureDict:
        desTarget = letterFeatureDict.get(letter)[1]

        matches = flanner.match(desTarget, desQuery)
        matches.sort(key=lambda x: x.distance)  # sort by distance
        numMatched = 0
        for i in range(len(matches)):
            if matches[i].distance > 50.0:
                break
            numMatched += 1
        matchedKPs.update({letter: numMatched})

    return matchedKPs


def getCannyEdge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cannyImg = cv2.Canny(gray, 100, 200)
    return cannyImg

def showCannyEdge(img):
    cannyImg = getCannyEdge(img)
    cv2.imshow("Canny", cannyImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getHoughLines(img):
    cannyImg = getCannyEdge(img)
    lines = cv2.HoughLinesP(cannyImg, 1, np.pi / 180,
                            threshold=5,
                            minLineLength=20, maxLineGap=10)
    for lineSet in lines:
        for line in lineSet:
            cv2.line(img, (line[0], line[1]), (line[2], line[3]),
                     (255, 255, 0))
    return img

def showHoughLines(img):
    getHoughLines(img)
    cv2.imshow("HoughLines", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def findFocus(img):
    # img = cv2.imread('hatch_with_background.png')

    blurImg = cv2.GaussianBlur(img, (11, 11), 0)

    edges = getCannyEdge(blurImg)
    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)  # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack



    # # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

    # Get the contour with the maximum area
    # maxAreaInd = 0
    # for i in range(len(contours)):
    #     area = cv2.contourArea(contours[i])
    #     if area > cv2.contourArea(contours[maxAreaInd]):
    #         maxAreaInd = i
    x, y, w, h = cv2.boundingRect(contours[0])

    # cv2.drawContours(queryImg, contours, -1, (0, 255, 0), 3)
    # hull = cv2.convexHull(contours)
    # cv2.imshow("contours", hull)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Draw the maximum area contour on the image
    # x, y, w, h = cv2.boundingRect(contours[maxAreaInd])

    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # # cv2.drawContours(queryImg, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('Contours', img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(x, y, w, h)
    return x, y, w, h

def drawFocus(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def matchImageTo(targetImgDir, queryImg):
    # queryImg = cv2.imread("hatch_with_background.png")
    x, y, w, h = findFocus(queryImg)

    # #DEBUG
    # cv2.imshow("queryImg", queryImg)
    # cv2.waitKey(0)
    #
    # cv2.imshow("roi", queryImg[y:y+h, x:x+w])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("roi.png", queryImg[y:y + h, x:x + w])
    roi = cv2.imread("roi.png")

    matchedKPs = getNumMatchedDict(targetImgDir, roi)
    sortedDic = sorted(matchedKPs, key=matchedKPs.get, reverse=True)
    targetImg = cv2.imread(targetImgDir + '/' + str(sortedDic[0]) + ".png")
    print(sortedDic)
    kpT, desT = computeORB(targetImg)
    kpQ, desQ = computeORB(roi)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)

    flanner = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flanner.match(desT, desQ)

    matches.sort(key=lambda x: x.distance)  # sort by distance
    i = 0
    for i in range(len(matches)):
        if matches[i].distance > 50.0:
            break

    roiToMatch = cv2.drawMatches(targetImg, kpT, roi, kpQ, matches[:i], None)

    cv2.imshow("Matches", roiToMatch)
    cv2.imshow("QueryImg", queryImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':


    queryImg = cv2.imread("hatch_with_background.png")
    matchImageTo('letterSamples', queryImg)

  
    # # use video
    # vidCap = cv2.VideoCapture(0)
    # while True:
    #     gotOne, img = vidCap.read()
    #     print(gotOne)
    #     if (gotOne):
    #         print("gotOne")
    #         x, y, w, h = findFocus(img)
    #         # print(img)
    #         # focusImg = img[y+h : y, x : x+w]
    #
    #         roi = img[x : x+w, y+h : y]
    #         cv2.imshow("roi", roi)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #
    #         print(x, y, w, h)
    #         # print(focusImg)
    #
    #         matchedKPs = getNumMatchedDict('letterSamples', roi)
    #
    #         sortedDic = sorted(matchedKPs, key=matchedKPs.get, reverse=True)
    #
    #         print(sortedDic[0])
    #
    #         print(matchedKPs)
    #
    #         targetImg = cv2.imread('letterSamples/'+str(sortedDic[0])+".png")
    #         kp, des = computeORB(targetImg)
    #         kp2, des2=computeORB(roi)
    #         FLANN_INDEX_LSH = 6
    #         index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                             table_number=6,  # 12
    #                             key_size=12,  # 20
    #                             multi_probe_level=1)  # 2
    #         search_params = dict(checks=50)
    #
    #         flanner = cv2.FlannBasedMatcher(index_params, search_params)
    #         matches = flanner.match(des, des2)
    #
    #         matches.sort(key=lambda x: x.distance)  # sort by distance
    #         i=0
    #         for i in range(len(matches)):
    #             if matches[i].distance > 50.0:
    #                 break
    #
    #         img3 = cv2.drawMatches(targetImg, kp, roi, kp2, matches[:i], None)
    #         img4 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
    #
    #         cv2.imshow("Matches", img4)
    #
    #     x = cv2.waitKey(10)  # Waiting may be needed for window updating
    #     char = chr(x & 0xFF)
    #     if (char == 'q'):  # esc == '27'
    #         break
    # cv2.destroyAllWindows()
    # vidCap.release()

