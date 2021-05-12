## Eye - Gaze Detector

## INTRODUCTION

An IoT device capable of detecting/traking the point where the eyes are looking within a boundary like laptop screen or monitor.

The detection is done using two main factors

  1. The pupil location : The possition of the pupil inside the eye itself is considered here. By  
     this the angle and the possition where the eye is looking at can be detected.
  2. The head pose : The head pose is also a major factor that changes accordinly with the eye gaze 
     movement so the direction of the focus can be tracked using the head pose.
     
Both of the above factors can be found in a cropped image of a face. But there will be many un wanted details which are not needed for the specific purpose of eye gaze tracking. so a mask is created and the gray scaled image of the eyes and the drawn lines representing the jaw border to indicate the head pose were displayed in the exact possition of the image on the mask. 

## HOW IT WORKS

The model runs on Raspberry Pi, the video feed taken by a Pi camera will be given as the input to a couple of classification models setted up in the device. The camera will be placed in the center of the top edge of the boundary (ie: laptop camera placement). The device attempts to  calculate the eye gaze location of the person looking the target screen or the monitor. The distance between the person and the camera should not be too short and should not be too long, the usual distance in practice can be considered as the best distance range.

## Install
```console
https://github.com/FOSSLife-foundation/Eye-gaze.git
```

*The models should be located and loaded before using.


## LANGUAGE

*Python - 100%.



