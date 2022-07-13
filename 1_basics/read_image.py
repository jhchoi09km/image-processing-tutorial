import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('test.jpg')
print('image size:', img.shape)
print('variable type:',type(img))
cv2.imshow('test image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

