import keras
import cv2
import numpy as np
import tensorflow

model = keras.models.load_model('model_1')

eye_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_eye.xml')
eye_dim = (70, 70)

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(img)
    
    if len(eyes):
        x, y, w, h = eyes[0]
        eye = img[y: y + h, x: x + h]
        eye = cv2.resize(eye, eye_dim, interpolation = cv2.INTER_AREA)
        prediction = model.predict(eye.reshape(1, eye_dim[0], eye_dim[1], 1))[0]
        print(prediction)
    else:
        prediction = [0, 0]
        print("no")
    img = cv2.circle(frame, (int(prediction[0]/10 * img.shape[1]), int(prediction[1]/10 * img.shape[0])), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.putText(frame,"x:" + str(prediction[0] )+ ",        y:" + str(prediction[1]), (120, 420), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.imshow("result", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
