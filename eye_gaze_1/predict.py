import keras
import cv2
import numpy as np

model = keras.models.load_model('model_0')

eye_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_eye.xml')
eye_dim = (70, 70)

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(img)
    
    try:
        x, y, w, h = eyes[0]
        eye = img[y: y + h, x: x + h]
        eye = cv2.resize(eye, eye_dim, interpolation = cv2.INTER_AREA)
        prediction = model.predict(eye.reshape(1, eye_dim[0], eye_dim[1], 1))
        print(prediction)
    except:
        print("no")
    
    if cv2.waitKey(5) & 0xFF == 27:
        break
