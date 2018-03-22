"""
@class: COMP380 AI Robotics
@date: 03/19/2018
@authors: Team SoJ (JJ Lim, So Jin Oh)
"""

import cv2
import numpy as np



def cmpTwoImgsUsing(compMethod, img1, img2):
    """
    Reads in two images. Make a grayscale copy of each image, and make a
    histogram for each picture from its grayscale version. Try each
    comparison method provided by compareHist.  How do the values returned
    correspond to image similarity?
    :return:
    """
    # cv2.HISTCMP_CORREL
    # cv2.HISTCMP_CHISQR
    # cv2.HISTCMP_CHISQR_ALT
    # cv2.HISTCMP_INTERSECT
    # cv2.HISTCMP_HELLINGER
    # cv2.HISTCMP_BHATTACHARYYA
    # cv2.HISTCMP_KL_DIV

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    compValue = cv2.compareHist(hist1, hist2, compMethod)

    return compValue

def printCmpMultImgsUsing(cmpMethod, img1, img2, img3, img4=None):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    hist3 = cv2.calcHist([gray3], [0], None, [256], [0, 256])

    cmpValList = []
    cmpValInds = []
    cmpVal_12 = cv2.compareHist(hist1, hist2, cmpMethod)
    cmpValList.append(cmpVal_12)
    cmpValInds.append("img1 and img2")
    cmpVal_13 = cv2.compareHist(hist1, hist3, cmpMethod)
    cmpValList.append(cmpVal_13)
    cmpValInds.append("img1 and img3")
    cmpVal_23 = cv2.compareHist(hist2, hist3, cmpMethod)
    cmpValList.append(cmpVal_23)
    cmpValInds.append("img2 and img3")
    if (img4 is not None):    #https://github.com/pandas-profiling/pandas-profiling/issues/61
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        hist4 = cv2.calcHist([gray4], [0], None, [256], [0, 256])
        cmpVal_14 = cv2.compareHist(hist1, hist4, cmpMethod)
        cmpValList.append(cmpVal_14)
        cmpValInds.append("img1 and img4")
        cmpVal_24 = cv2.compareHist(hist2, hist4, cmpMethod)
        cmpValList.append(cmpVal_24)
        cmpValInds.append("img2 and img4")
        cmpVal_34 = cv2.compareHist(hist3, hist4, cmpMethod)
        cmpValList.append(cmpVal_34)
        cmpValInds.append("img3 and img4")

    cmpMethodStr = ""
    # # cv2.HISTCMP_CORREL
    # # cv2.HISTCMP_CHISQR
    # # cv2.HISTCMP_CHISQR_ALT
    # # cv2.HISTCMP_INTERSECT
    # # cv2.HISTCMP_HELLINGER
    # # cv2.HISTCMP_BHATTACHARYYA
    # # cv2.HISTCMP_KL_DIV
    if cmpMethod == 0:
        cmpMethodStr = "HISTCMP_CORREL"
    elif cmpMethod == 1:
        cmpMethodStr = "HISTCMP_CHISQR"
    elif cmpMethod == 2:
        cmpMethodStr = "HISTCMP_CHISQR_ALT"
    elif cmpMethod == 3:
        cmpMethodStr = "HISTCMP_INTERSECT"
    elif cmpMethod == 4:
        cmpMethodStr = "HISTCMP_HELLINGER"
    elif cmpMethod == 5:
        cmpMethodStr = "HISTCMP_BHATTACHARYYA"
    elif cmpMethod == 6:
        cmpMethodStr = "HISTCMP_KL_DIV"

    for cmpVal,cmpInd in zip(cmpValList,cmpValInds): #https://stackoverflow.com/questions/1663807/how-to-iterate-through-two-lists-in-parallel/43062307#43062307
        print("Histogram similarity between " + cmpInd + " using " + cmpMethodStr + " is " + str(cmpVal))


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

def getHoughCircles(img):
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # im2, contrs, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Hough Circles", origIm)

    cannyImg = getCannyEdge(img)
    houghCircles = cv2.HoughCircles(cannyImg, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=50, param2=30,
                                    minRadius=10, maxRadius=50)
    for circleSet in houghCircles:
        for circle in circleSet:    #(x, y, radius) .
            cv2.circle(img, (circle[0], circle[1]), circle[2], (255, 255, 0))


    return img

def showHoughCircles(img):
    getHoughCircles(img)
    toStr = "HoughCircles for " + str(img)
    cv2.imshow(toStr, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showVidImgsWithCannyAndHough():
    vidCap = cv2.VideoCapture(0)
    while True:
        ret, img = vidCap.read()
        # cv2.imshow("Webcam", img)
        cannyImg = getCannyEdge(img)
        cv2.imshow("Canny Edges", cannyImg)
        houghImg = getHoughLines(img)
        cv2.imshow("Hough Lines", houghImg)

        x = cv2.waitKey(10)  # Waiting may be needed for window updating
        char = chr(x & 0xFF)
        if (char == 'q'):   #esc == '27'
            break
    cv2.destroyAllWindows()
    vidCap.release()




if __name__ == '__main__':
    # histograms for a grayscale img with different histSize (bin size)
    # img = cv2.imread("TestImages/shops.jpg")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hist256 = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # hist8 = cv2.calcHist([gray], [0], None, [8], [0, 256])
    # print(hist256)
    # print(hist8)

    # back-projections for matching imgs: Each pixel in the image is
    # compared to the histogram, and its value is based on the height
    # of the histogram for that value.
    # img1 = cv2.imread("TestImages/SnowLeo3.jpg")
    # img2 = cv2.imread("TestImages/SnowLeo2.jpg")
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # hist = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    # bp = cv2.calcBackProject([gray2], [0], hist, [0, 256], 1)
    # cv2.imshow("BackProject", bp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # comparing histograms for image similarity
    # img1 = cv2.imread("TestImages/SnowLeo3.jpg")
    # img2 = cv2.imread("TestImages/SnowLeo2.jpg")
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    # hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    # compValue = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # # cv2.HISTCMP_CORREL
    # # cv2.HISTCMP_CHISQR
    # # cv2.HISTCMP_CHISQR_ALT
    # # cv2.HISTCMP_INTERSECT
    # # cv2.HISTCMP_HELLINGER
    # # cv2.HISTCMP_BHATTACHARYYA
    # # cv2.HISTCMP_KL_DIV
    # print("Histogram similarity is " + str(compValue))
    # print("Histogram similarity is " + str(cmpTwoImgs(img1, img2, cv2.HISTCMP_CORREL)))

    img1 = cv2.imread("TestImages/SnowLeo3.jpg")
    img2 = cv2.imread("TestImages/SnowLeo2.jpg")
    img3 = cv2.imread("TestImages/beachBahamas.jpg")
    img4 = cv2.imread("TestImages/gorge.jpg")

    """ MILESTONE 1: Comparing Images"""
    # print("##### 3 imgs vs. 4 imgs #####")
    # printCmpMultImgsUsing(cv2.HISTCMP_CORREL, img1, img2, img3)
    # print()
    # printCmpMultImgsUsing(cv2.HISTCMP_CORREL, img1, img2, img3, img4)
    #
    # print()
    #
    # print("##### Different Testing Methods #####")
    # printCmpMultImgsUsing(cv2.HISTCMP_CORREL, img1, img2, img3)
    # print()
    # printCmpMultImgsUsing(cv2.HISTCMP_CHISQR, img1, img2, img3)
    # print()
    # printCmpMultImgsUsing(cv2.HISTCMP_CHISQR_ALT, img1, img2, img3)
    # print()
    # printCmpMultImgsUsing(cv2.HISTCMP_INTERSECT, img1, img2, img3)
    # print()
    # printCmpMultImgsUsing(cv2.HISTCMP_HELLINGER, img1, img2, img3)
    # print()
    # printCmpMultImgsUsing(cv2.HISTCMP_BHATTACHARYYA, img1, img2, img3)
    # print()
    # printCmpMultImgsUsing(cv2.HISTCMP_KL_DIV, img1, img2, img3)
    # print()


    """ MILESTONE 2: Experimenting with Canny and Hough Functions """
    img5 = cv2.imread("TestImages/shops.jpg")
    # showSobelGradient(img5)

    # showCannyEdge(img5)

    img6 = cv2.imread("TestImages/Puzzle1.jpg")

    # showHoughLines(img6)

    # showVidImgsWithCannyAndHough()

    """ MILESTONE 3: Coins and puzzles and line detection """
    coinIm1 = cv2.imread('TestImages/Coins1.jpg')
    coinIm2 = cv2.imread('TestImages/Coins2.jpg')
    showHoughCircles(coinIm1)
    showHoughCircles(coinIm2)

    # imgray = cv2.cvtColor(origIm1, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # # im2, contrs, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.imshow("Hough Circles", origIm)
    #
    # cannyImg = getCannyEdge(origIm1)
    # houghCircles = cv2.HoughCircles(cannyImg, cv2.HOUGH_GRADIENT, 1, 20,
    #                                 param1=50, param2=30,
    #                                 minRadius=10, maxRadius=50)
    #
    # for circleSet in houghCircles:
    #     for circle in circleSet:    #(x, y, radius) .
    #         cv2.circle(origIm1, (circle[0], circle[1]), circle[2], (255, 255, 0))

    # cv2.imshow("Hough Circles", origIm1)
    # # cv2.drawContours(origIm, origIm1, -1, (0, 255, 0), 3)
    # # cv2.imshow('Contours', origIm)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

