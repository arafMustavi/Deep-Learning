import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('reference.jpg',0)
img2 = img.copy()



cap = cv2.VideoCapture('input.mov')
while cap.isOpened():
    isSuccess, template = cap.read()
    img = img.astype(np.float32)
    template = template.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = img.shape
    # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR','cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    img = img2.copy()
    method = cv2.TM_CCOEFF
    res = cv2.matchTemplate(template,img,cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #     top_left = min_loc
    # else:
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    # gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Input Video',template)
    cv2.imshow('Output Video',img)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# template = cv.imread('template.jpg',0)
# All the 6 methods for comparison in a list
