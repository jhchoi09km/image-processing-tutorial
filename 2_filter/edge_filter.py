import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('bird.jpg')
kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

img_blur = cv2.filter2D(img, -1, kernel)

cv2.imshow('original', img)
cv2.imshow('blurred', img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

