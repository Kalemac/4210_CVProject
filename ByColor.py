# Python program for Detection of a
# specific color(blue here) using OpenCV with Python
import cv2
import numpy as np

# Get the frame
frame = cv2.imread(r'images/game1.jpg', cv2.IMREAD_UNCHANGED)

# Converts images from BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([150, 255, 255])


mask = cv2.inRange(hsv, lower_blue, upper_blue)


res = cv2.bitwise_and(frame, frame, mask=mask)

blueCont = cv2.findContours(mask.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
blueCont = blueCont[0] if len(blueCont) == 2 else blueCont[1]

for blue in blueCont:
    x, y, w, h = cv2.boundingRect(blue)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
cv2.waitKey()