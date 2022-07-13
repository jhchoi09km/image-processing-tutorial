import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('test.jpg')
img_rect = cv2.rectangle(img, (480,250), (680,450), (0,0,255), 5)


cv2.imshow('rect', img_rect)
cv2.waitKey(0)
cv2.destroyAllWindows()

