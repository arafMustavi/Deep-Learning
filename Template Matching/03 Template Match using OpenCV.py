import cv2
import numpy as np
from matplotlib import pyplot as plt

imgReference = cv2.imread('customeReference.jpg', 0)
# cv2.imshow('Reference Image', imgReference)
refImageSummary = imgReference.shape
print(refImageSummary)
key = cv2.waitKey(0)
h,w = imgReference.shape

imgTemplate = cv2.imread('FirstFrame.jpg', 0)
# cv2.imshow('Template Image', imgTemplate)
tempImageSummary = imgTemplate.shape
print(tempImageSummary)
key = cv2.waitKey(0)


res = cv2.matchTemplate(imgReference,imgTemplate,cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(imgTemplate, top_left, bottom_right, (255,0,0), 2)
cv2.imshow('Marked Image', imgTemplate)

# print(res)