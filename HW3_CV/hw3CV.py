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


def getBFTargetCluster(targetImgsDir):
    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    letterFeatureDict = readLetterFeatures(targetImgsDir)
    for letter in letterFeatureDict:
        desTarget = letterFeatureDict.get(letter)[1]
        bfMatcher.add(desTarget)
    return bfMatcher

def getNumMatchedDict(targetImgsDir, queryImg):
    # queryImg = cv2.imread(queryImg)

    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,  # 12
    #                     key_size=12,  # 20
    #                     multi_probe_level=1)  # 2
    # search_params = dict(checks=50)

    # flanner = cv2.FlannBasedMatcher(index_params, search_params)

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
        print(matches)
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



    size = 11

    blurImg = cv2.GaussianBlur(img, (size, size), 0)
    # cv2.imshow("blur", blurImg) #DEBUG
    # edges = getCannyEdge(blurImg)
    edges = getCannyEdge(img)
    # showCannyEdge(img)
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
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        return x, y, w, h
    else:
        return None

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
    # return x, y, w, h

def drawFocus(img, x, y, w, h):
    if x is None:
        return None
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def findRectangleROI(queryImg):
    edges = getCannyEdge(queryImg)

    st = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    res = cv2.morphologyEx(src=edges, op=cv2.MORPH_CLOSE, kernel=st)  #

    _, contours, h = cv2.findContours(res, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)  # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True),
                                  True)  # https://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
        # print(len(approx))
        if (len(approx) == 4):
            # cv2.drawContours(queryImg, [cnt], 0, (0, 0, 255), -1)
            x, y, w, h = cv2.boundingRect(cnt)
            # x, y, w, h = cv2.boundingRect(cnt.x)
            # print(x, y, w, h)
    roi = queryImg[y:y + h, x:x + w]
    # cv2.imshow('roi', roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return roi

def matchImageTo(targetImgDir, queryImg):
    # queryImg = cv2.imread("hatch_with_background.png")
    roi = findRectangleROI(queryImg)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    orb = cv2.ORB_create(nfeatures=15)

    kpQuery, desQuery = orb.detectAndCompute(roi, None)

    letterFeatureDict = readLetterFeatures(targetImgDir)
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

    sortedDic = sorted(matchedKPs, key=matchedKPs.get, reverse=True)
    targetImg1 = cv2.imread(targetImgDir + '/' + str(sortedDic[0]) + ".png")
    targetImg2 = cv2.imread(targetImgDir + '/' + str(sortedDic[1]) + ".png")

    cv2.imshow("Matched #1", targetImg1)
    cv2.imshow("Matched #2", targetImg2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(sortedDic)

    # bfMatcher = getBFTargetCluster(targetImgDir)
    #
    #
    # # matchedKPs = getNumMatchedDict(targetImgDir, roi)
    # # sortedDic = sorted(matchedKPs, key=matchedKPs.get, reverse=True)
    # # targetImg = cv2.imread(targetImgDir + '/' + str(sortedDic[0]) + ".png")
    # # print(sortedDic)
    # # kpT, desT = computeORB(targetImg)
    # kpQ, desQ = computeORB(roi)
    #
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,  # 12
    #                     key_size=12,  # 20
    #                     multi_probe_level=1)  # 2
    # search_params = dict(checks=50)
    #
    # # flanner = cv2.FlannBasedMatcher(index_params, search_params)
    # # matches = flanner.match(desT, desQ)
    # matches = bfMatcher.match(desQ, None)
    #
    # matches.sort(key=lambda x: x.distance)  # sort by distance
    # # i = 0
    # # for i in range(len(matches)):
    # #     if matches[i].distance > 50.0:
    # #         break
    #
    # for i in range(len(matches)):
    #     print(matches[i].ingIdx)
    # # roiToMatch = cv2.drawMatches(targetImg, kpT, roi, kpQ, matches[:i], None)
    #
    # # return roiToMatch
    # # cv2.imshow("Matches", roiToMatch)
    # # cv2.imshow("QueryImg", queryImg)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)


    queryImg = cv2.imread("hatch_with_background.png")    #11
    # queryImg = cv2.imread("b_with_background.png")    #13
    # queryImg = cv2.imread("257_with_background.png")
    #
    #
    #
    # edges = getCannyEdge(queryImg)
    #
    # # showCannyEdge(queryImg)
    # st = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #
    #
    # res = cv2.morphologyEx(src=edges, op=cv2.MORPH_CLOSE,kernel=st) #
    #
    #
    # _, contours, h = cv2.findContours(res, cv2.RETR_TREE,
    #                                   cv2.CHAIN_APPROX_SIMPLE)  # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
    #
    # for cnt in contours:
    #     approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True),True)   # https://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
    #     # print(len(approx))
    #     if (len(approx) == 4):
    #         # cv2.drawContours(queryImg, [cnt], 0, (0, 0, 255), -1)
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         # x, y, w, h = cv2.boundingRect(cnt.x)
    #         # print(x, y, w, h)
    roi = findRectangleROI(queryImg)
    cv2.imwrite("roi.png", roi)
    # cv2.imshow("roi", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    # size = 5
    #
    # blurImg = cv2.GaussianBlur(queryImg, (size, size), 0)

    # gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("blur", blurImg) #DEBUG
    # edges = getCannyEdge(blurImg)
    # edges = getCannyEdge(blurImg)
    # showCannyEdge(blurImg)
    # showCannyEdge(img)
    # ret, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    # _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
    #                                   cv2.CHAIN_APPROX_SIMPLE)  # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
    #
    # # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html


    # x, y, w, h = 0,0,0,0
    # if len(contours) > 0:
    #     x, y, w, h = cv2.boundingRect(contours[2])
    #     # return x, y, w, h
    # # else:
    #     # return None
    #
    # cv2.imwrite("roi.png", queryImg[y:y + h, x:x + w])
    # roi = cv2.imread("roi.png")


    matchImageTo('letterSamples', queryImg)

    # gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)
    #
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,  # 12
    #                     key_size=12,  # 20
    #                     multi_probe_level=1)  # 2
    # search_params = dict(checks=50)
    #
    # flanner = cv2.FlannBasedMatcher(index_params, search_params)
    #
    #
    # orb = cv2.ORB_create(nfeatures=15)
    # """"""
    # kpQuery, desQuery = orb.detectAndCompute(roi, None)
    #
    # letterFeatureDict = readLetterFeatures('letterSamples')
    # matchedKPs = {}
    # for letter in letterFeatureDict:
    #     desTarget = letterFeatureDict.get(letter)[1]
    #
    #     matches = bf.match(desQuery, desTarget)
    #     matches = sorted(matches, key=lambda x: x.distance)
    #     numMatched = 0
    #     for i in range(len(matches)):
    #         if matches[i].distance > 50.0:
    #             break
    #         numMatched += 1
    #     matchedKPs.update({letter: numMatched})
    #
    # sortedDic = sorted(matchedKPs, key=matchedKPs.get, reverse=True)
    # targetImg = cv2.imread('letterSamples' + '/' + str(sortedDic[0]) + ".png")
    # print(sortedDic)






    # Since, we have index of only one training image,
    # all matches will have imgIdx set to 0.







    # matchImageTo('letterSamples', queryImg)
    # cv2.imshow("Matching", matchImageTo('letterSamples', queryImg))
    # cv2.imshow("Query Image", queryImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # #use video
    # vidCap = cv2.VideoCapture(0)
    # while True:
    #     gotOne, img = vidCap.read()
    #     print(gotOne)
    #     if (gotOne):
    #         print("gotOne")
    #         if not matchImageTo('letterSamples', img) is None:
    #             cv2.imshow("Matching", matchImageTo('letterSamples', img))
    #             cv2.imshow("VideoCam", img)
    #         else:
    #             # cv2.imshow("Matching", matchImageTo('letterSamples', img))
    #             cv2.imshow("VideoCam", img)
    #     x = cv2.waitKey(10)  # Waiting may be needed for window updating
    #     char = chr(x & 0xFF)
    #     if (char == 'q'):  # esc == '27'
    #         break
    # cv2.destroyAllWindows()
    # vidCap.release()