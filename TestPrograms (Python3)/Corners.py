import cv2
import numpy as np

img = cv2.imread("TestImages/Puzzle1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = np.float32(gray)

# Compute Harris
dst = cv2.cornerHarris(gray2, 2, 3, 0.04)

cv2.imshow("HarrisImage", dst)

# Isolate Harris corners from image
dilDst = dst
dilDst = cv2.dilate(dst, None)
thresh = 0.1 * dst.max()
ret, threshDst =  cv2.threshold(dilDst, thresh, 255, cv2.THRESH_BINARY)


# Display corners points
disp = np.uint8(threshDst)
cv2.imshow("Harris", disp)


# Compute Harris
goodFeats = cv2.goodFeaturesToTrack(gray, 1000, 0.1, 5)

for row in goodFeats:
    x = row[0, 0]
    y = row[0, 1]
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

cv2.imshow("Shi-Tomasi", img)


cv2.waitKey(0)
cv2.destroyAllWindows()