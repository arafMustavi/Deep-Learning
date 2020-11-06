import cv2
import numpy as np
from matplotlib import pyplot as plt

imgReference = cv2.imread('reference.jpg', 0)
cv2.imshow('Reference Image', imgReference)
refImageSummary = imgReference.shape

print(imgReference)
# greyFrame = cv2.cvtColor(imgReference, cv2.COLOR_BGR2GRAY)
#
# cv2.imwrite('greyReference.jpg', greyFrame)
imgReference = cv2.imread('FirstFrame.jpg', 0)
cv2.imshow('Reference Image', imgReference)
refImageSummary = imgReference.shape
print(imgReference)
