import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('FirstFrame.jpg',0)
# img2 = img.copy()
template = cv.imread('reference.jpg',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img1 = img.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img1,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + 100, top_left[1] + 100)
    cv.rectangle(img1,top_left, bottom_right, (255,0,0), 2)
    cv.imshow("Res",img1)
    key = cv.waitKey(0)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img1,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()