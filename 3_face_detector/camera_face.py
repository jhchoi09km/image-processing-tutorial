import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
    x,y,w,h = faces_rect[0]

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv2.imshow('video', img)
    cv2.waitKey(100)


cam.release()
cv2.destroyAllWindows()

