import cv2 as cv
import numpy as np

img1 = cv.imread('ml.png')
img2 = cv.imread('opencv-logo.png')
dst = cv.addWeighted(img1,0.7,img2,0.3,0)

while(1):
    cv.imshow('dst',dst)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cv.destroyAllWindows()