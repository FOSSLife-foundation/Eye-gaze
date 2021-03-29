import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor= dlib.shape_predictor("/home/andrew/project/scratch/shape_predictor_68_face_landmarks.dat")

#taking vdo feed from webcam
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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
         
        if (y1e<y2e):
            cv2.rectangle(frame, (x1e-15, y1e-15), (x2e+15,y2e+15), (0,255,0), 1)
        else:
            cv2.rectangle(frame, (x1e-15, y1e+15), (x2e+15,y2e-15), (0,255,0), 1)
            
        yaw_pts= np.array( [ [yaw1.x,yaw1.y],[yaw2.x,yaw2.y],[yaw3.x,yaw3.y],[yaw4.x,yaw4.y],[yaw5.x,yaw5.y],[yaw6.x,yaw6.y],[yaw7.x,yaw7.y] ], np.int32 )
        #yaw_pts = yaw_pts.reshape((-1,1,2))
        cv2.polylines(frame, [yaw_pts], True, (50,175,0), 1)
        
        leye_pts = np.array( [ [leye1.x,leye1.y],[leye2.x,leye2.y],[leye3.x,leye3.y],[leye4.x,leye4.y],[leye5.x,leye5.y] ,[leye6.x,leye6.y] ], np.int32 )
        #leye_pts = leye_pts.reshape((-1,1,2))
        #cv2.polylines(frame, [leye_pts], True, (0,0,255),2)
        
        reye_pts = np.array( [ [reye1.x,reye1.y],[reye2.x,reye2.y],[reye3.x,reye3.y],[reye4.x,reye4.y],[reye5.x,reye5.y] ,[reye6.x, reye6.y]], np.int32 )
        #reye_pts = reye_pts.reshape((-1,1,2))
        #cv2.polylines(frame, [reye_pts], True, (0,0,255),2)
        
        height , width ,_ =  frame.shape
        mask_l = np.zeros( ( height, width), np.uint8)
        mask_r = np.zeros( ( height, width), np.uint8)
        
        #drawing eye polygon om mask
        cv2.polylines(mask_r, [reye_pts], True, 255, 2)
        cv2.polylines(mask_l, [leye_pts], True, 255, 2)
        cv2.fillPoly(mask_l,  [leye_pts], 255)
        cv2.fillPoly(mask_r,  [reye_pts], 255)
        
        left_eye = cv2.bitwise_and(gray, gray, mask = mask_l)
        right_eye =cv2.bitwise_and(gray, gray, mask = mask_r)
        
        
        
        #cropping left eyes
        min_le_x = np.min(leye_pts[ :, 0 ])
        max_le_x = np.max(leye_pts[ :, 0 ])
        min_le_y = np.min(leye_pts[ :, 1 ])
        max_le_y = np.max(leye_pts[ :, 1])
        
        left_eye = left_eye [min_le_y:max_le_y, min_le_x: max_le_x]
        left_eye =cv2.resize(left_eye, None, fx=5, fy=5)
        cv2.imshow("Left_eye", left_eye)
        
        #cropping right eyes
        min_re_x = np.min(reye_pts[ :, 0 ])
        max_re_x = np.max(reye_pts[ :, 0 ])
        min_re_y = np.min(reye_pts[ :, 1 ])
        max_re_y = np.max(reye_pts[ :, 1])
        
        right_eye = right_eye [min_re_y:max_re_y, min_re_x: max_re_x]
        right_eye =cv2.resize(right_eye, None, fx=5, fy=5)
        cv2.imshow("right_eye", right_eye)
        
        #gray_left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        #gray_right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("rightt_eye", gray_right_eye)
        #cv2.imshow("Left_eye", gray_left_eye)
        
        _, threshold_leye = cv2.threshold(left_eye, 70 , 255 , cv2.THRESH_BINARY)
        _, threshold_reye = cv2.threshold(right_eye, 70 , 255 , cv2.THRESH_BINARY)
        
        #threshold_leye = cv2.resize(threshold_leye, None, fx=5, fy =5)
        #threshold_reye = cv2.resize(threshold_reye, None, fx=5, fy =5)
        cv2.imshow("Threshold_left" , threshold_leye)
        cv2.imshow("Threshold_right" , threshold_reye)
        cv2.imshow("Mask_l", mask_l)
        cv2.imshow("Mask_r" ,mask_r)
        
    cv2.imshow("Frame", frame)
    
    #x = input("press enter to continue: " )
    
    
    key = cv2.waitKey(1)
    if key ==27:
        break
        
cap.release()
cv2.destroyAllWindows()
