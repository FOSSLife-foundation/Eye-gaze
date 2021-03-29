import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("/home/andrew/project/scratchhaarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier("/home/andrew/project/scratchhaarcascade_eye.xml")

cap = cv2.VideoCapture(0)
while True:
     ret, img = cap.read()
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
     count=1
     for (x,y,w,h) in faces:
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = img[y:y+h, x:x+w]

         eyes = eye_cascade.detectMultiScale(roi_gray)
         for (ex,ey,ew,eh) in eyes:
             crop_img = roi_color[ey: ey + eh, ex: ex + ew]
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
             s="{0}.jpg"
             s1='/home/andrew/Pictures/Webcam/'+s.format(count)
             count=count+1
             cv2.imwrite(s1,crop_img)
     cv2.imshow('img',img)
     k = cv2.waitKey(30) & 0xff
     if k == 27:
         break

cap.release()
cv2.destroyAllWindows()