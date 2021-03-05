from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cv2
import numpy as np

IMAGE_PATH = "/media/andrew/New Volume/studies/4th sem/project/eye_gaze-20210128T125925Z-001/eye_gaze/dataset/"

grid_w = 10
grid_h = 10

face_cascade = cv2.CascadeClassifier('/home/andrew/project/scratchhaarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/media/andrew/New Volume/studies/4th sem/project/eye_gaze-20210128T125925Z-001/haarcascade_eye.xml')
eye_dim = (70, 70)

for i in range(grid_w):
 for j in range(grid_h):
  img = cv2.imread(IMAGE_PATH + str(i) + '.' + str(j) , cv2.IMREAD_GRAYSCALE)
  #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = face_cascade.detectMultiScale(img, 1.3, 5)
  for (x,y,w,h) in faces:
   img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
   roi_gray = img[y:y+h, x:x+w]
   roi_color = img[y:y+h, x:x+w]
   eyes = eye_cascade.detectMultiScale(roi_gray)
   for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  cv2.imshow('img',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
