#!/usr/bin/env python3
"""
@class: COMP380 AI Robotics
@date: 03/20/2018
@authors: Team SoJ (JJ Lim, So Jin Oh)

This is for Homework 3: Computer Vision Task2. Reads a video frame, extracts a
region of interest, and (attempts to) detect a word. Displays two best matched words
out of a directory of target words.
"""

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import time
import pandas as pd

def initORB(numFeatures=25):
    """
    Initiate STAR detection (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) (https://stackoverflow.com/questions/32702433/opencv-orb-detector-finds-very-few-keypoints)
    :param numFeatures: the number of features that will be detected and computed using ORB
    :return: initiated orb
    """
    return cv2.ORB_create(nfeatures=numFeatures, scoreType=cv2.ORB_FAST_SCORE)

def showORB(img, imgKeypoints):
    """
    Display ORB Keypoints on the image
    """

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

#TODO: Fix this
def getNumMatchedDict(targetImgsDir, queryImg):
    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    letterFeatureDict = readLetterFeatures(targetImgsDir)

    # kpQuery, desQuery = computeORB(queryImg)
    orb = cv2.ORB_create()
    kpQuery, desQuery = orb.detectAndCompute(queryImg, None)



    matchedKPs = {}
    for letter in letterFeatureDict:
        desTarget = letterFeatureDict.get(letter)[1]

        # matches = flanner.match(desTarget, desQuery)
        matches = bfMatcher.match(desTarget, desQuery)

        matches.sort(key=lambda x: x.distance)  # sort by distance
        # print(matches) # DEBUG
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


def drawFocus(img, x, y, w, h):
    if x is None:
        return None
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def findRectangleROI(queryImg):
    """
    Finds a rectangle in the query image that will be used as the region of interest. Preprocessing
    before matchImageTo.
    :param queryImg:
    :return:
    """
    edges = getCannyEdge(queryImg)

    st = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    res = cv2.morphologyEx(src=edges, op=cv2.MORPH_CLOSE, kernel=st)  #

    _, contours, h = cv2.findContours(res, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)  # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)  # https://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
        if (len(approx) == 4):
            x, y, w, h = cv2.boundingRect(cnt)


    # print("x, y, w, h:", x, y, w, h) #DEBUG
    if (x == 0 and y == 0 and w == 0 and h == 0):
        print("ROI: No (queryImg)")
        return queryImg
    roi = queryImg[y:y + h, x:x + w]
    cv2.imwrite('roi.png', roi)

    print("ROI: Yes")
    return roi

def matchImageTo(targetImgDir, queryImg):
    #Set up Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #Set up ORB
    # orb = cv2.ORB_create(nfeatures=15)
    orb = cv2.ORB_create()

    # Get the keypoints and descriptors of the preprocessed region of interest in the query image
    roi = findRectangleROI(queryImg)
    # DEBUG
    # cv2.imshow("roi", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    kpQuery, desQuery = orb.detectAndCompute(roi, None)

    # Get the dictionary that has kp,des of each image in the target image directory
    letterFeatureDict = readLetterFeatures(targetImgDir)

    # number of matches for each image will be stored with the name of the image (without extension) as a key value
    matchedKPs = {}
    for letter in letterFeatureDict:
        desTarget = letterFeatureDict.get(letter)[1]
        matches = bf.match(desQuery, desTarget)
        matches = sorted(matches, key=lambda x: x.distance)
        numMatched = 0
        for i in range(len(matches)):
            if matches[i].distance > 50.0:
                break
            numMatched += 1
        matchedKPs.update({letter: numMatched})
    # matchedKPs = getNumMatchedDict(targetImgDir, roi)
    #sort the matchedKPs to get first two best matches
    sortedDic = sorted(matchedKPs, key=matchedKPs.get, reverse=True)

    targetImg1 = cv2.imread(targetImgDir + '/' + str(sortedDic[0]) + ".png")
    kpTarget1, desTarget1 = letterFeatureDict.get(sortedDic[0])
    numMatch1 = matchedKPs.get(sortedDic[0]) # if numMatch1 == 0 --> just dont do...

    targetImg2 = cv2.imread(targetImgDir + '/' + str(sortedDic[1]) + ".png")
    kpTarget2, desTarget2 = letterFeatureDict.get(sortedDic[1])
    numMatch2 = matchedKPs.get(sortedDic[1])

    # #DEBUG: display first two best matches
    # cv2.imshow("Matched #1", targetImg1)
    # cv2.imshow("Matched #2", targetImg2)
    # cv2.waitKey(0)


    #draw matches with first two best matches
    matches1 = bf.match(desQuery, desTarget1)
    # Check for error in drawMatches
    isExecutable1 = True
    for m in range(len(matches1)):
        i1 = matches1[m].queryIdx
        i2 = matches1[m].trainIdx
        if i1 >= len(kpTarget1):
            isExecutable1 = False
        if i2 >= len(kpQuery):
            isExecutable1 = False

    if not isExecutable1:
        roiToMatch1 = roi
        print("roiToMatch1: roi")
    else:
        roiToMatch1 = cv2.drawMatches(targetImg1, kpTarget1, roi.copy(), kpQuery, matches1[:numMatch1], None)
        print("roiToMatch1: drawMatches")

    matches2 = bf.match(desQuery, desTarget2)
    isExecutable2 = True
    for m in range(len(matches2)):
        i1 = matches2[m].queryIdx
        i2 = matches2[m].trainIdx
        if i1 >= len(kpTarget2):
            isExecutable2 = False
        if i2 >= len(kpQuery):
            isExecutable2 = False
    if not isExecutable2:
        roiToMatch2 = roi
        print("roiToMatch2: roi")

    else:
        roiToMatch2 = cv2.drawMatches(targetImg2, kpTarget2, roi.copy(), kpQuery, matches2[:numMatch2], None)
        print("roiToMatch2: drawMatches")

    return sortedDic, roiToMatch1, roiToMatch2


if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)

    # # use an image
    # print("first")
    # queryImg = cv2.imread("hatch_with_background.png")    #11
    # print("here")
    # sortedDict, roiToMatch1, roiToMatch2 = matchImageTo("letterSamples",queryImg)
    # cv2.imshow("roiToMatch1", roiToMatch1)
    # cv2.imshow("roiToMatch2", roiToMatch2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("there")


    df = pd.DataFrame() # https://stackoverflow.com/questions/16597265/appending-to-an-empty-data-frame-in-pandas

    # use video frames
    vidCap = cv2.VideoCapture(0)


    while True:
        start_time = time.time()
        gotOne, frame = vidCap.read()

        if (gotOne):
            print("--------------------------------------------------------------------------")
            frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
            cv2.imshow("VideoCam", frame)
            sortedDict, roiToMatch1, roiToMatch2 = matchImageTo("letterSamples", frame)

            # print("sortedDict, roiToMatch1, roiToMatch2:", sortedDict, roiToMatch1, roiToMatch2) # DEBUG
            time.sleep(1.0 - time.time() + start_time)  # Sleep for 1 second minus elapsed time https://stackoverflow.com/questions/48525971/processing-frame-every-second-in-opencv-python

            if (sortedDict is not None):
                log = {'match1': sortedDict[0], 'match2': sortedDict[1], 'match3': sortedDict[2]}
                df = df.append(log, ignore_index=True)

                print(sortedDict)
                cv2.namedWindow("roiToMatch1")
                cv2.moveWindow("roiToMatch1", 0, 400)
                cv2.namedWindow("roiToMatch2")
                cv2.moveWindow("roiToMatch2", 300, 400)
                cv2.imshow("roiToMatch1", roiToMatch1)
                cv2.imshow("roiToMatch2", roiToMatch2)
            else:
                # time.sleep(1.0)  # Sleep for 1 second minus elapsed time
                pass

                # if not matchImageTo('letterSamples', frame) is None:
            #     print("not none")
            #     cv2.imshow("Matching", matchImageTo('letterSamples', frame))
            #     cv2.imshow("VideoCam", frame)
            # else:
            #     print("none")
            #     cv2.imshow("VideoCam", frame)
        x = cv2.waitKey(10)  # Waiting may be needed for window updating
        char = chr(x & 0xFF)
        if (char == 'q'):  # esc == '27'
            break
        filename = 'HW3_task_2_b_match_results.csv'
        df.to_csv(filename, sep=',', encoding='utf-8', index=False)

    cv2.destroyAllWindows()
    vidCap.release()


