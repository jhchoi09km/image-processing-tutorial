import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('test.jpg')
print('center pixel values:',img[960//2, 540//2, :])

crop = img[250:451,480:681, :]
print('cropped image size:',crop.shape)

flip = img[::-1,:,:]

cv2.imshow('test crop', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('flip', flip)
cv2.waitKey(0)
cv2.destroyAllWindows()
