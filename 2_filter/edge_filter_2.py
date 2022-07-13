import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('bird.jpg')
kernel = -np.ones((5,5))
kernel[2,2] = 25

img_blur = cv2.filter2D(img, -1, kernel)

cv2.imshow('original', img)
cv2.imshow('blurred', img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

