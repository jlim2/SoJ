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

def readImgFeatures(img):
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
    orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
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
    # orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
    orb = cv2.ORB_create(nfeatures=20, scoreType=cv2.ORB_FAST_SCORE)

    # keypoints = orb.detect(image=img)
    keypoints, des = orb.detectAndCompute(img, None)

    return keypoints, des

def readLetterFeatures(dir):
    # imgNames = []
    # for img in alphabetImgs:
    #     imgName = img.strip(".png")
    #     imgNames.append(imgName)
    # print(imgNames)

    imgFiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    # testImgFiles.append([f for f in listdir("TestImages/Letters") if isfile(join("TestImages/Letters", f))])

    # imgNames = []
    letterFeatureDict = {}
    for imgFile in imgFiles:
        if (imgFile.endswith('.png')):
            # imgNames.append(imgName)
            img = cv2.imread("letterSamples/" + imgFile)
            # print(type(imgFile))
            # print(imgFile)
            kp, des = computeORB(img)
            # print(kp)
            # print(des)
            imgName = imgFile.strip(".png")
            # alphFeatureDict.update({imgFile: (kp, des)})
            letterFeatureDict.update({imgName: (kp, des)})


    return letterFeatureDict

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



    # imgA = cv2.imread('alphabets/A.png')
    # showORB(imgA)
    # keypoints, descriptors = computeORB(imgA)
    # print(keypoints)
    # print(readModelAlphabets('alphabets'))
    # print(des.shape)
    # print(kp)
    # print(des)

    #Set up Flanner
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)

    flanner = cv2.FlannBasedMatcher(index_params, search_params)

    letterFeatureDict = readLetterFeatures('letterSamples')
    letterImgs = letterFeatureDict.keys()

    targetImgs = {}
    for img in letterImgs:
        targetImg = cv2.imread("letterSamples/"+img+".png")

        kpTarget = letterFeatureDict.get(img)[0]
        desTarget = letterFeatureDict.get(img)[1]

    #targetImg = cv2.imread('alphabets/A.png')
    #kpTarget = alphFeatureDict.get('A')[0]
    #desTarget = alphFeatureDict.get('A')[1]




    # vidCap = cv2.VideoCapture(0)
    # while True:
    #     gotOne, frame = vidCap.read()
    #     if (gotOne):
    #         print("gotOne")
    #         kpQuery, desQuery = computeORB(frame)
    #
    #         # for targetImg in alphImgs:
    #
    #         matches = flanner.match(desTarget, desQuery)
    #         matches.sort(key=lambda x: x.distance)  # sort by distance
    #
    #
    #         matchedKPs = {}
    #         # draw matches with distance less than threshold
    #         #i = 0
    #         #compare each alphImgs to frame and count how many matches
    #         # for img in alphImgs:
    #         #     desTarget = alphFeatureDict.get(img)[1]
    #         #     matches = flanner.match(desTarget, desQuery)
    #         #     for i in range(len(matches)):
    #         #         if matches[i].distance > 50.0:
    #         #             break
    #         #         print(matches[i])
    #
    #
    #         for alph in letterFeatureDict:
    #             targetImg = cv2.imread('letterSamples/'+alph+'.png')
    #             kpTarget = letterFeatureDict.get(alph)[0]
    #             desTarget = letterFeatureDict.get(alph)[1]
    #
    #             matches = flanner.match(desTarget, desQuery)
    #             matches.sort(key=lambda x: x.distance)  # sort by distance
    #             numMatched = 0
    #             for i in range(len(matches)):
    #                 if matches[i].distance > 50.0:
    #                     break
    #                 numMatched += 1
    #             matchedKPs.update({alph: numMatched})
    #
    #             # print("matches[i]: "+str(matches[i]), end=" ")
    #             # print("distance: "+str(matches[i].distance), end=" ")
    #             # print("imgIdx: "+str(matches[i].imgIdx), end=" ")
    #             # print("queryIdx: " + str(matches[i].queryIdx), end=" ")
    #             # print("trainIdx: " + str(matches[i].trainIdx))
    #             # queryImg_idx = matches[i].queryIdx
    #             # targetImg_idx = matches[i].trainIdx
    #             #
    #             # (x1, y1) = kpQuery[queryImg_idx].pt
    #             # (x2, y2) = kpTarget[targetImg_idx].pt
    #
    #             img3 = cv2.drawMatches(targetImg, kpTarget, frame, kpQuery, matches[:i], None)
    #             img4 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
    #             cv2.imshow("Matches", img4)
    #         print(numMatched)
    #
    #         #if numMatched >= 22:
    #         #    print("matched!")
    #
    #
    #
    #         x = cv2.waitKey(20)
    #         c = chr(x & 0xFF)
    #         if c == 'q':
    #             break
    #
    # vidCap.release()
    # cv2.destroyAllWindows()

    frame = cv2.imread('letterSamples/hatch.png')
    kpQuery, desQuery = computeORB(frame)

    matchedKPs = {}
    i = 0
    for letter in letterFeatureDict:
        targetImg = cv2.imread('letterSamples/' + letter + '.png')
        kpTarget = letterFeatureDict.get(letter)[0]
        desTarget = letterFeatureDict.get(letter)[1]

        matches = flanner.match(desTarget, desQuery)
        matches.sort(key=lambda x: x.distance)  # sort by distance
        numMatched = 0
        for i in range(len(matches)):
            if matches[i].distance > 50.0:
                break
            numMatched += 1
        matchedKPs.update({letter: numMatched})

    #     img3 = cv2.drawMatches(targetImg, kpTarget, frame, kpQuery, matches[:i], None)
    #     img4 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
    #     cv2.imshow("Matches", img4)
    #     x = cv2.waitKey(20)
    #     c = chr(x & 0xFF)
    #     if c == 'q':
    #         break
    # cv2.destroyAllWindows()

    # print(matchedKPs)
    d_sorted_by_value = OrderedDict(sorted(matchedKPs.items(), key=lambda x: x[1]))
    for k, v in d_sorted_by_value.items():
        print("%s: %s" % (k, v))
