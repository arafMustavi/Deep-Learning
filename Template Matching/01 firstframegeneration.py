import cv2
import numpy as np
from matplotlib import pyplot as plt

# LOAD the Reference Image
imgReference = cv2.imread('reference.jpg', 0)
cv2.imshow('Reference Image', imgReference)
refImageSummary = imgReference.shape
print(refImageSummary)
# PRINTS the TUPLE for THE REFERENCE IMAGE (98,54) which is HEIGHT and WIDTH
key = cv2.waitKey(0)

totalFrameCount = 0
firstFrame = None

# LOAD the INPUT VIDEO
cap = cv2.VideoCapture('input.mov')
while cap.isOpened():
    isSuccess, frame = cap.read()

    if not isSuccess:
        break

    totalFrameCount += 1

    # CONVERTS into GRAYSCALE
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if firstFrame is None:
        firstFrame = greyFrame
        cv2.imwrite('FirstFrame.jpg', firstFrame)
    frameSummary = greyFrame.shape
    print(frameSummary)
    # PRINTS the TUPLE for THE FRAME IMAGE (413,433,3) which is HEIGHT and WIDTH and COLOR CHANNEL
    # UPON GRAYSCALE CONVERSION THE FRAME IMAGE TURNS INTO (413,433)

    cv2.imshow('Input Video',greyFrame)
    key = cv2.waitKey(1)
    # key = cv2.waitKey(0) Will pause the video for eternity
    # key = cv2.waitKey(1) Will create 1 ms Delay
    if key == ord('q'):
        break

print("Total FRAME Rendered",totalFrameCount)

cap.release()
cv2.destroyAllWindows()
# template = cv.imread('template.jpg',0)
# All the 6 methods for comparison in a list
