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
import os

def identifyWords(img):
    """First, identify if a sign is in an image by looking (perhaps) for a white or black box, or a box of
    the given background color). Next, use the affine or perspective transformations to warp the image to
    scale, center, and orient the discovered box straight on.
    """

    pass



def showSobelGradient(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute gradient in horizontal direction (detects vertical edges)
    sobelValsHorz = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    horzImg = cv2.convertScaleAbs(sobelValsHorz)
    cv2.imshow("horizontal gradient", horzImg)

    # Compute gradient in vertical direction (Detects horizontal edges)
    sobelValsVerts = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    vertImg = cv2.convertScaleAbs(sobelValsVerts)
    cv2.imshow("vertical gradient", vertImg)

    # Combine the two gradients
    sobelComb = cv2.addWeighted(sobelValsHorz, 0.5,
                                sobelValsVerts, 0.5, 0)
    # Convert back to uint8
    sobelImg = cv2.convertScaleAbs(sobelComb)
    cv2.imshow("Sobel", sobelImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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


def showORB(img):
    # Initiate STAR detection (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) (https://stackoverflow.com/questions/32702433/opencv-orb-detector-finds-very-few-keypoints)
    orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
    keypoints = orb.detect(image=img)

    img2 = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    cv2.imshow("ORB KeyPoints", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def computeORB(img):
    """
    An image that has been read
    :param img:
    :return: keypoints, des
    """
    # Initiate STAR detection (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) (https://stackoverflow.com/questions/32702433/opencv-orb-detector-finds-very-few-keypoints)
    orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
    keypoints = orb.detect(image=img)
    keypoints, des = orb.compute(img, keypoints)

    return keypoints, des

def readModelAlphabets(alphabetImgs):
    alphFeatureDict = {}
    # imgNames = []
    # for img in alphabetImgs:
    #     imgName = img.strip(".png")
    #     imgNames.append(imgName)
    # print(imgNames)

    imgFiles = [f for f in listdir("alphabets") if isfile(join("alphabets", f))]
    # testImgFiles.append([f for f in listdir("TestImages/Letters") if isfile(join("TestImages/Letters", f))])

    # imgNames = []
    alphFeatureDict = {}
    for imgFile in imgFiles:
        if (imgFile.endswith('.png')):
            imgName = imgFile.strip(".png")
            # imgNames.append(imgName)
            img = cv2.imread(imgFile)
            # print(imgFile)
            kp, des = computeORB(img)
            print(kp)
            print(des)
            alphFeatureDict.update({imgName: (kp, des)})

    return alphFeatureDict

if __name__ == '__main__':
    img = cv2.imread('LetterInWhite.png', 0)
    # showCannyEdge(img)
    # cannyEdges = getCannyEdge(img)
    # for edge in cannyEdges:
    #     print(edge)
    # print(cannyEdges)

    # Initiate STAR detection (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) (https://stackoverflow.com/questions/32702433/opencv-orb-detector-finds-very-few-keypoints)
    # orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
    # keypoints = orb.detect(image=img)
    #
    #
    # img2 = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))
    # cv2.imshow("ORB KeyPoints", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #
    # showORB(img)
    # keypoints, descriptors = computeORB(img)
    # for kp in keypoints:
    #     print(str(kp.pt))
    # print(descriptors.shape)
    # for des in descriptors:
    #     pass
        # print(des.shape)


    print(readModelAlphabets('alphabets'))
    imgA = cv2.imread('alphabets/A.png')
    showORB(imgA)
    keypoints, descriptors = computeORB(imgA)
    # print(des.shape)
    # print(kp)
    # print(des)