import cv2
import numpy as np

cam = cv2.VideoCapture(0)
interval = 1

while True:
    _, img = cam.read()
    img = cv2.resize(img, dsize=(0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    if len(faces_rect)<1:
        cv2.imshow('video', img)
        cv2.waitKey(interval)
        continue

    for i in range(len(faces_rect)):
        x,y,w,h = faces_rect[i]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    
    cv2.imshow('video', img)
    cv2.waitKey(interval)


cam.release()
cv2.destroyAllWindows()

