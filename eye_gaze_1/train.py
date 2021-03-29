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

labels = []
images = []

eye_cascade = cv2.CascadeClassifier('/media/andrew/New Volume/studies/4th sem/project/eye_gaze-20210128T125925Z-001/haarcascade_eye.xml')
eye_dim = (70, 70)

for i in range(grid_w):
    for j in range(grid_h):
        label = [i, j]
        img = cv2.imread(IMAGE_PATH + str(i) + '.' + str(j), cv2.IMREAD_GRAYSCALE)
        eyes = eye_cascade.detectMultiScale(img)
	
        try:
            x, y, w, h = eyes[0]
            eye = img[y: y + h, x: x + h]
            eye = cv2.resize(eye, eye_dim, interpolation = cv2.INTER_AREA)
            labels.append(label)
            images.append(eye.reshape(eye_dim[0], eye_dim[1], 1))
        except:
            print(label, "not detected")

images = np.array(images)
labels = np.array(labels)

print(images.shape)
i=0
while True:
    cv2.imshow("eye", images[i])
    i = i+1
    if cv2.waitKey(50) & 0xFF == 27:
        break
        

#xtrain, xtest, ytrain, ytest=train_test_split(images, labels, test_size=0.15) 

#model = Sequential()
#model.add(Conv2D(16, (3,3), activation='relu', input_shape = (eye_dim[0], eye_dim[1], 1)))
#model.add(MaxPooling2D(2, 2))
#model.add(Conv2D(32, (3,3), activation='relu'))
#model.add(MaxPooling2D(2, 2))
#model.add(Flatten())
#model.add(Dense(64, activation="relu"))
#model.add(Dense(2))
#model.compile(loss="mse", optimizer="adam")

#model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=1)

#model.save("model_1")


