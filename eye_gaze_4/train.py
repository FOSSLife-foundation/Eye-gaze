from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

root_path = "/home/andrew/project/eye gaze/dataset/"
grid_w = 10
grid_h = 10

labels = []
images = []

eye_dim = (70, 70)

for i in range(grid_w):
    for j in range(grid_h):
        fld_nm = str(i) +"," + str(j)
        label = [i, j]
        for fl_nm in os.listdir( root_path + fld_nm):
            try:
                img = cv2.imread(root_path + fld_nm + "/" + fl_nm , cv2.IMREAD_GRAYSCALE )
                if img is not None:
                    img = cv2.resize(img, eye_dim, interpolation = cv2.INTER_AREA)
                    img = img.reshape(70, 70, 1)
                    images.append(img)
                    #images.append(img.reshape(70, 70, 1))
                    labels.append(label)
                    #cv2.imshow("eye",  img)
            except:
                print(label, "not detected")
	

images = np.array(images)
labels = np.array(labels)

#print(images.shape)
#i=0
#while True:
 #   cv2.imshow("eye", images[i])
  #  i = i+1
   # if cv2.waitKey(50) & 0xFF == 27:
    #    break
        

xtrain, xtest, ytrain, ytest=train_test_split(images, labels, test_size=0.15) 

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape = (eye_dim[0], eye_dim[1], 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(2))
model.compile(loss="mse", optimizer="adam")

model.fit(xtrain, ytrain, batch_size=12, epochs=200, verbose=1 )

model.save("model_0")


