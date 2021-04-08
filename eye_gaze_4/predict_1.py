import keras
import cv2
import numpy as np
#import tensorflow
import dlib
import os
import sys

model = keras.models.load_model('model_0')

eye_dim = ( 70, 70 )

root_path = '/home/andrew/project/eye gaze/'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor ( root_path + "shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    
    if not success:
        break
    
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #eyes = eye_cascade.detectMultiScale(img)
    _, frame = cap.read()
    gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:
        x,y =face.left(), face.top()
        x1,y1 = face.right() , face.bottom()
        #cv2.rectangle(frame, (x,y), (x1,y1) , (0,155,1) ,1)
        
        landmarks = predictor(gray, face)
        
        x1e  = landmarks.part(36).x
        y1e  = landmarks.part(36).y
        x2e  = landmarks.part(45).x
        y2e  = landmarks.part(45).y
        
        yaw1 = landmarks.part(3)
        yaw2 = landmarks.part(5)
        yaw3 = landmarks.part(7)
        yaw4 = landmarks.part(8)
        yaw5 = landmarks.part(9)
        yaw6 = landmarks.part(11)
        yaw7 = landmarks.part(13)
        
        leye1 = landmarks.part(36)
        leye2 = landmarks.part(37)
        leye3 = landmarks.part(38)
        leye4 = landmarks.part(39)
        leye5 = landmarks.part(40)
        leye6 = landmarks.part(41)
        
        reye1 = landmarks.part(42)
        reye2 = landmarks.part(43)
        reye3 = landmarks.part(44)
        reye4 = landmarks.part(45)
        reye5 = landmarks.part(46)
        reye6 = landmarks.part(47)
         
        #if (y1e<y2e):
         #   cv2.rectangle(frame, (x1e-15, y1e-15), (x2e+15,y2e+15), (0,255,0), 1)
        #else:
         #   cv2.rectangle(frame, (x1e-15, y1e+15), (x2e+15,y2e-15), (0,255,0), 1)
     
        yaw_pts= np.array( [ [yaw1.x,yaw1.y],[yaw2.x,yaw2.y],[yaw3.x,yaw3.y],[yaw4.x,yaw4.y],[yaw5.x,yaw5.y],[yaw6.x,yaw6.y],[yaw7.x,yaw7.y] ], np.int32 )
        #yaw_pts = yaw_pts.reshape((-1,1,2))
       
        leye_pts = np.array( [ [leye1.x,leye1.y],[leye2.x,leye2.y],[leye3.x,leye3.y],[leye4.x,leye4.y],[leye5.x,leye5.y] ,[leye6.x,leye6.y] ], np.int32 )
        #leye_pts = leye_pts.reshape((-1,1,2))
         
        reye_pts = np.array( [ [reye1.x,reye1.y],[reye2.x,reye2.y],[reye3.x,reye3.y],[reye4.x,reye4.y],[reye5.x,reye5.y] ,[reye6.x, reye6.y]], np.int32 )
        #reye_pts = reye_pts.reshape((-1,1,2))
        
        height , width ,_ =  frame.shape
        mask_l = np.zeros( ( height, width), np.uint8)

        #drawing eye polygon om mask
        cv2.polylines(mask_l, [reye_pts], True, 255, 1)
        cv2.polylines(mask_l, [leye_pts], True, 255, 1)
        cv2.fillPoly(mask_l,  [leye_pts], 255)
        cv2.fillPoly(mask_l,  [reye_pts], 255)
        
        sample = cv2.bitwise_and(gray, gray, mask = mask_l)
        cv2.polylines(sample, [yaw_pts], True, (50,175,0), 3)
        #cv2.polylines(sample, [reye_pts], True, (0,0,255),2)
        #cv2.polylines(sample, [leye_pts], True, (0,0,255),2)
        sample = sample[y+15:y1+15, x:x1]
        #sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        sample = cv2.resize(sample, eye_dim, interpolation = cv2.INTER_AREA)
        sample = sample.reshape(70, 70, 1)
    
        if len(sample):
            prediction = model.predict(sample.reshape(1, 70, 70, 1))[0]
            print(prediction)
        else:
            prediction = [0, 0]
            print("no")
        #img = cv2.circle(frame, (int(prediction[0]/10 * gray.shape[1]), int(prediction[1]/10 * gray.shape[0] )), radius=5, color=(0, 0, 255), thickness=-1)
        #cv2.putText(frame,"x:" + str(prediction[0] )+ ",        y:" + str(prediction[1]), (120, 420), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        if ( 1.5 < int(prediction[0]) < 4.5 ) & (2 < int(prediction[1]) < 6 ) :
            cv2.putText(frame, " CENTER" ,  (230, 250), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
        elif ( int(prediction[0]) < 3 ) & (int(prediction[1]) < 4 ) :
            cv2.putText(frame, " LEFT_UP" ,  (55, 64), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
        elif ( int(prediction[0]) > 3 ) & (int(prediction[1]) < 4 ) :
            cv2.putText(frame, "LEFT_DOWN" ,  (70, 380), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
        elif (int(prediction[0]) < 3 ) & (int(prediction[1]) > 4) :
            cv2.putText(frame, " RIGHT_UP" ,  (475, 82), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
        elif (int(prediction[0]) > 3 ) & (int(prediction[1]) > 4) :
            cv2.putText(frame, " RIGHT_DOWN" ,  (400, 400), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("result", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            sys.exit()
            

