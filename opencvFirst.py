import cv2
import numpy as np
# Making a list of files in a directory using os module (https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory)
from os import listdir
from os.path import isfile, join
# OpenCV Documentation https://docs.opencv.org/3.0-last-rst/


# img1 = cv2.imread("TestImages/SnowLeo1.jpg")
# cv2.imshow("Leopard 1", img1)
# img2 = cv2.imread("TestImages/SnowLeo2.jpg")
# cv2.imshow("Leopard 2", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



""" Milestone 1: Making a slideshow """
# Milestone 1 - phase 1
imgsForSlideShow = []
imgsForSlideShow.append(cv2.imread("TestImages/SnowLeo1.jpg"))
imgsForSlideShow.append(cv2.imread("TestImages/SnowLeo2.jpg"))
imgsForSlideShow.append(cv2.imread("TestImages/chicago.jpg"))
imgsForSlideShow.append(cv2.imread("TestImages/garden.jpg"))

for i in range (4):
    cv2.imshow("Slide Show", imgsForSlideShow[i])
    cv2.waitKey(0)
    if i==3:
        cv2.destroyAllWindows()

""" Milestone 2: Making a slideshow of all pictures in a folder """
testImgFiles = [f for f in listdir("TestImages") if isfile(join("TestImages", f))]
#testImgFiles.append([f for f in listdir("TestImages/Letters") if isfile(join("TestImages/Letters", f))])

fileIndex = 0
for imgFile in testImgFiles:
    if imgFile.endswith('jpg') or imgFile.endswith('jpeg') or imgFile.endswith('JPG'):
        imgName = "TestImages img number " + str(fileIndex)
        imgFile = "TestImages/" + imgFile
        print(imgFile)
        img = cv2.imread(imgFile)
        cv2.imshow(imgName, img)
        cv2.waitKey(0)
        #cv2.waitKey(fileIndex)
        print(fileIndex, len(testImgFiles))
        if (fileIndex == len(testImgFiles)-2):
            print("HERE")
            cv2.destroyAllWindows()
        fileIndex += 1

""" Milestone 3: Marking up images """
leo = cv2.imread("TestImages/SnowLeo2.jpg")
cv2.circle(leo, (120, 120), 60, (180, 180, 20), 3)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(leo, "SnowLeo2.jpg", (20, 30), font, 0.7, (255, 255, 255))
cv2.rectangle(leo, (240, 120), (550, 300), (180, 180, 20), -1)
cv2.imshow("Leopard 2", leo)
cv2.waitKey(0)
cv2.destroyAllWindows()


""" Drawing and shaving shapes """
# The following two statements make blank images colored black (draw1)
# or white (draw2)
draw1 = np.zeros((300, 500, 3), np.uint8)
draw2 = 255 * np.ones((500, 300, 3), np.uint8)

cv2.line(draw2, (50, 50), (150, 250), (0, 0, 255))
cv2.rectangle(draw1, (10, 100), (100, 10), (0, 180, 0), -1)
cv2.circle(draw2, (30, 30), 30, (220, 0, 0), -1)
cv2.ellipse(draw1, (250, 150), (100, 60), 30, 0, 220, (250, 180, 110), -1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(draw1, "Hi, there", (10, 270), font, 1, (255, 255, 255))

cv2.imshow("Black", draw1)
cv2.imshow("White", draw2)

cv2.imwrite("blackPic.jpg", draw1)
cv2.imwrite("whitePic.jpg", draw2)

cv2.waitKey(0)
cv2.destroyAllWindows()


"""
OpenCV drawing commands

cv2.line(img, pt1, pt2, col)
    Takes in an image, and two points given as tuples that specify pixels (col, row), and a
    color, and draws a line on the image between those two points.

cv2.rectangle(img, pt1, pt2, col)
    Takes in an image, and two points in the image, and a color, and draws a rectangle with one point as upper-left
    corner and one as lower-right.

cv2.circle(img, pt, rad, col)

cv2.ellipse(img, pt,
            axes, angle,
            startAng, endAng,
            col)

    Takes in an image and inputs that specify an ellipse or elliptical arc. The point is the center of the ellipse. Axes
    is also a tuple containing the length of major and minor axes. Angle indicates rotation of ellipse around center.
    Start and ending angles are how much of the ellipse to draw.

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, text, pt,
            font, fSize, col)
    Takes in an image, a string, a point for lower-left corner of text, a font and font size, and a color, and draws
    text as specified. Start font size at 1!

Colors here should be tuples containing three values (blue, green, red). Each value should be between 0 and 255
Most commands take an optional line-thickness input. When negative, it causes a filled shape to be drawn.


"""

""" Get images using built-in camera """
vidCap = cv2.VideoCapture(0)
for i in range(300):
    ret, img = vidCap.read()
    cv2.imshow("Webcam", img)
    cv2.waitKey(10)   # Waiting may be needed for window updating

cv2.destroyAllWindows()
vidCap.release()


"""
In-built Camera Usage

cap = cv2.VideoCapture(0)
    Creates a VideoCapture object connected to specified camera

cap.isOpened()
    Boolean method returns true if camera connection succeeded

ret, image = cap.read()
    Method returns two values, a code to tell if the image was read successfully, and an image from the video stream

cap.release()
    Method disconnects from camera

"""