import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor= dlib.shape_predictor("/home/andrew/project/scratch/shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = detector(gray)
    for face in faces:
        x,y =face.left(), face.top()
        x1,y1 = face.right() , face.bottom()
        #cv2.rectangle(frame, (x,y), (x1,y1) , (0,155,1) ,1)
        
        landmarks = predictor(gray, face)
        
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

        
        leye_pts = np.array( [ [leye1.x,leye1.y],[leye2.x,leye2.y],[leye3.x,leye3.y],[leye4.x,leye4.y],[leye5.x,leye5.y] ,[leye6.x,leye6.y] ], np.int32 )
        #leye_pts = leye_pts.reshape((-1,1,2))
        #cv2.polylines(frame, [leye_pts], True, (0,0,255),2)
        
        reye_pts = np.array( [ [reye1.x,reye1.y],[reye2.x,reye2.y],[reye3.x,reye3.y],[reye4.x,reye4.y],[reye5.x,reye5.y] ,[reye6.x, reye6.y]], np.int32 )
        #reye_pts = reye_pts.reshape((-1,1,2))
        #cv2.polylines(frame, [reye_pts], True, (0,0,255),2)
        
        #height , width ,_ =  frame.shape
        #mask_l = np.zeros( ( height, width), np.uint8)
        #mask_r = np.zeros( ( height, width), np.uint8)
        #cv2.imshow("ml" , mask_l)
        #cv2.imshow("mr", mask_r)
        
        #drawing eye polygon om mask
        #cv2.polylines(mask_r, [reye_pts], True, 255, 2)
        #cv2.polylines(mask_l, [leye_pts], True, 255, 2)
        #cv2.fillPoly(mask_l,  [leye_pts], 255 )
        #cv2.fillPoly(mask_r,  [reye_pts], 255 )
        
        #left_eye = cv2.bitwise_and (gray, gray, mask = mask_l)
        #right_eye =cv2.bitwise_and (gray, gray, mask = mask_r)
        
        
        
        #cropping left eyes
        min_le_x = np.min(leye_pts[ :, 0 ])
        max_le_x = np.max(leye_pts[ :, 0 ])
        min_le_y = np.min(leye_pts[ :, 1 ])
        max_le_y = np.max(leye_pts[ :, 1])
        
        left_eye = gray [min_le_y:max_le_y, min_le_x: max_le_x]
        left_eye =cv2.resize(left_eye, None, fx=5, fy=5)
        #left_eye= cv2.GaussianBlur(left_eye,(5,5) ,0)
        cv2.imshow("Left_eye", left_eye)
        
        #cropping right eyes
        min_re_x = np.min(reye_pts[ :, 0 ])
        max_re_x = np.max(reye_pts[ :, 0 ])
        min_re_y = np.min(reye_pts[ :, 1 ])
        max_re_y = np.max(reye_pts[ :, 1])
        
        right_eye = gray [min_re_y:max_re_y, min_re_x: max_re_x]
        right_eye =cv2.resize(right_eye, None, fx=5, fy=5)
        #right_eye= cv2.GaussianBlur(right_eye,(5,5) ,0)
        cv2.imshow("right_eye", right_eye)
        
        #gray_left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        #gray_right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("rightt_eye", gray_right_eye)
        #cv2.imshow("Left_eye", gray_left_eye)
        
        _, threshold_leye = cv2.threshold(left_eye, 60 , 255 , cv2.THRESH_BINARY)
        _, threshold_reye = cv2.threshold(right_eye, 60 , 255 , cv2.THRESH_BINARY)
        
        #threshold_leye = cv2.resize(threshold_leye, None, fx=5, fy =5)
        #threshold_reye = cv2.resize(threshold_reye, None, fx=5, fy =5)
        cv2.imshow("Threshold_left" , threshold_leye)
        cv2.imshow("Threshold_right" , threshold_reye)
        

        

    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(30)
    if key == 27:
        break
        
     
        
cv2.destroyAllWindows()

