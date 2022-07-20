import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    chk, frame = cam.read()
    cv2.imshow('video', frame)
    cv2.waitKey(1)


cam.release()
cv2.destroyAllWindows()

