import cv2
import numpy as np
from matplotlib import pyplot as plt

imgReference = cv2.imread('reference.jpg', 0)
# cv2.imshow('Reference Image', imgReference)
refImageSummary = imgReference.shape
print(refImageSummary)
# PRINTS the TUPLE for THE REFERENCE IMAGE (98,54) which is HEIGHT and WIDTH
key = cv2.waitKey(0)

delta_frame = cv2.absdiff(imgReference, imgReference)
print(cv2.sumElems(delta_frame)[0])




# imgTemplate = cv2.imread('FirstFrame.jpg', 0)
imgTemplate = cv2.imread('FirstFrame.jpg', 0)
# cv2.imshow('Template Image', imgTemplate)
tempImageSummary = imgTemplate.shape
print(tempImageSummary)
# PRINTS the TUPLE for THE REFERENCE IMAGE (413,433) which is HEIGHT and WIDTH
key = cv2.waitKey(0)

crop_img = imgTemplate[0:98, 0:54]
crop_imgSummary = crop_img.shape
print(crop_imgSummary)
delta_frame = cv2.absdiff(crop_img, imgReference)
print(cv2.sumElems(delta_frame)[0])
