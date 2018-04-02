"""
@authors: Susan E. Fox

This is for Macalester's CS course COMP380 AI Robotics
"""

import cv2
import numpy as np

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)

# Some versions of OpenCV need this to fix a bug
cv2.ocl.setUseOpenCL(False)

targetImg = cv2.imread("/Users/JJ/SoJ/HW3_CV/hatch_with_background.png")
# targetImg = cv2.imread("AyoubSudoku.jpg")
cv2.imshow("target", targetImg)
# create an ORB object, that can run the ORB algorithm.
orb = cv2.ORB_create()  # some versions use cv2.ORB_create() instead

bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
flanner = cv2.FlannBasedMatcher(index_params, search_params)

kp1, des1 = orb.detectAndCompute(targetImg, None)

cap = cv2.VideoCapture(0)

while True:
    gotOne, frame = cap.read()
    if gotOne:
        kp2, des2 = orb.detectAndCompute(frame, None)

        matches = bfMatcher.match(des1, des2)
        # matches = flanner.match(des1, des2)
        matches.sort(key = lambda x: x.distance)  # sort by distance

        # draw matches with distance less than threshold
        i = 0
        for i in range(len(matches)):
            if matches[i].distance > 50.0:
                break
        print(i)
        img3 = cv2.drawMatches(targetImg, kp1, frame, kp2, matches[:i], None)
        img4 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Matches", img4)
    x = cv2.waitKey(20)
    c = chr(x & 0xFF)
    if c == 'q':
        break

cap.release()
cv2.destroyAllWindows()