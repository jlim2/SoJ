"""
@class: COMP380 AI Robotics
@date: 03/07/2018
@authors: Team SoJ (JJ Lim, So Jin Oh)
"""

import cv2
import numpy as np
import random

def numpyBasics():
    image = cv2.imread("TestImages/SnowLeo4.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray image", gray)
    # cv2.waitKey(0)
    blackImg = np.zeros((150, 250), np.uint8)
    cv2.imshow("Blank image", blackImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    type(image)
    Out[1]: numpy.ndarray
    type(gray)
    Out[2]: numpy.ndarray
    type(blackImg)
    Out[3]: numpy.ndarray

    '''A tuple giving the dimensions of the array (height, width)
    for grayscale images, and (height, width, depth) for color images'''
    image.shape
    Out[4]: (177, 284, 3)
    gray.shape
    Out[5]: (177, 284)
    blackImg.shape
    Out[6]: (150, 250)

    '''The total number of pixels in the image'''
    image.size
    Out[8]: 150804
    gray.size
    Out[9]: 50268
    blackImg.size
    Out[10]: 37500


    '''The type of number used in the array'''
    image.dtype
    Out[11]: dtype('uint8')
    gray.dtype
    Out[12]: dtype('uint8')
    blackImg.dtype
    Out[14]: dtype('uint8')

    """
def colorBGR():
    image = cv2.imread("TestImages/shops.jpg")
    (bc, gc, rc) = cv2.split(image)

    cv2.imshow("Blue channel", bc)
    cv2.imshow("Green channel", gc)
    cv2.imshow("Red channel", rc)
    cv2.waitKey(0)

    zc = np.zeros(bc.shape, np.uint8) #zero color
    blueImg = cv2.merge((bc, zc, zc))
    cv2.imshow("Blue channel", blueImg)
    greenImg = cv2.merge((zc, gc, zc))
    cv2.imshow("Green channel", greenImg)
    redImg = cv2.merge((zc, zc, rc))
    cv2.imshow("Red channel", redImg)
    cv2.waitKey(0)


def colorHSV():
    """
    OpenCV uses the range from 0 to 180, and divides every typical hue value by 2.  For example, yellow shades in the
    diagram run from 45 to 60. In OpenCV the corresponding hues would be 22 to 30.  Notice that red includes values from
    both ends of the range.

    HSV is useful for color-tracking in images, because the hue changes relatively little when the brightness of the
    light changes, whereas all three channels in an RGB image change when the brightness changes.

    :return:
    """
    pass

def funColorEffect(img):
    """
    Define a script that reads in a color image. It should split the channels of the image, and convert the resulting
    tuple into a list. Then, it should put the color channels back together in a different order.  If you want to try
    something new, import the random module, make a list of the three channels, call the random.shuffle function and
    pass it the list of channel arrays. It will randomly reorder the elements in the list. Finally, make a new image by
    merging the shuffled channels together.

    :param img: img read with cv2.imread
    :return: img with funner color
    """

    (bc, gc, rc) = cv2.split(img)

    cv2.imshow("Blue channel", bc)
    cv2.waitKey(0)
    cv2.imshow("Green channel", gc)
    cv2.waitKey(0)
    cv2.imshow("Red channel", rc)
    cv2.waitKey(0)


    bgrList = [bc, gc, rc]

    bgrImg = cv2.merge((bgrList[0], bgrList[1], bgrList[2]))
    rgbImg = cv2.merge((bgrList[2], bgrList[1], bgrList[0]))
    grbImg = cv2.merge((bgrList[1], bgrList[2], bgrList[0]))

    cv2.imshow("Original BGR Image", bgrImg)
    cv2.waitKey(0)

    while True:
        random.shuffle(bgrList)
        #print(bgrList)
        newImg = cv2.merge((bgrList[0], bgrList[1], bgrList[2]))
        cv2.imshow("Color Reorganized Image", newImg)
        code = cv2.waitKey(0)
        char = chr(code & 0xFF)
        if char == 'q':
            break
        #cv2.imshow("Color Reorganized GRB Image", grbImg)
    cv2.destroyAllWindows()

def blendImg(image1, image2):
    """
    Blends two image evenly at first. Then, as the user hits a key, the weight of the image1 gets larger until the
    image2 completely fades out.
    :param image1:
    :param image2:
    :return:
    """
    weight = 0.5
    while weight < 1:
        blendedImage = cv2.addWeighted(image1, weight, image2, 1 - weight, 0)
        cv2.imshow("Blended Image", blendedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        weight += 0.1

def rotatingImg(img, num=100):
    """
    a program to rotate an image repeatedly if the user hits a key.
    :param img:
    :param num:
    :return:
    """
    cv2.imshow("Original", img)
    cv2.waitKey(0)
    (rows, cols, depth) = img.shape
    rotAng = 0
    while True:
        rotMat = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotAng, 1)
        rotAng += 15
        rotImg = cv2.warpAffine(img, rotMat, (cols, rows))
        cv2.imshow("Rotated", rotImg)
        # if chr(cv2.waitKey(0) & 0xFF) == "q":
        #     break
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    #numpyBasics()
    #colorBGR()
    image = cv2.imread("TestImages/shops.jpg")
    #funColorEffect(image)
    #rotatingImg(image)


    image1 = cv2.imread("TestImages/redDoor.jpg")
    image2 = cv2.imread("TestImages/garden.jpg")
    blendImg(image1, image2)

