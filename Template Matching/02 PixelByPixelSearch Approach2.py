import cv2
import numpy as np
from matplotlib import pyplot as plt

imgReference = cv2.imread('customeReference.jpg', 0)
# cv2.imshow('Reference Image', imgReference)
refImageSummary = imgReference.shape
print(refImageSummary)
# PRINTS the TUPLE for THE REFERENCE IMAGE (98,54) which is HEIGHT and WIDTH
key = cv2.waitKey(0)

imgTemplate = cv2.imread('FirstFrame.jpg', 0)
# imgTemplate = cv2.imread('Test.jpg', 0)
# cv2.imshow('Template Image', imgTemplate)
tempImageSummary = imgTemplate.shape
print(tempImageSummary)
# PRINTS the TUPLE for THE REFERENCE IMAGE (413,433) which is HEIGHT and WIDTH
key = cv2.waitKey(0)
min = 10000000000
# for y in range(209 - 98 + 1):
#     for x in range(113 - 54 + 1):
for y in range(413 - 125 + 1):
    for x in range(433 - 118 + 1):
        crop_img = imgTemplate[y:y + 125, x:x + 118]
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        delta_frame = cv2.multiply(crop_img,imgReference)
        # delta_frame = cv2.absdiff(crop_img, imgReference)
        # print(y,x)
        cv2.imshow("Ghost Frame", delta_frame)
        intensity_delta = cv2.sumElems(delta_frame)[0]
        # print(intensity_delta)
        if intensity_delta < min:
            min = intensity_delta
            print("Current Min", min)
            print(y, x)

        if intensity_delta == 0.0:
            print("It's a Match!")
            break

        # print(delta_frame)
        # print(cv2.sumElems(delta_frame))
        key = cv2.waitKey(1)

# for i in range(0,413):
print(min)


# Current Min 7339.0
# 228 104