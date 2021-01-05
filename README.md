INTRODUCTION

An IoT device capable of detecting the distance of the eyes from a certain point and detecting the point where the eyes are looking within a boundary. 

HOW IT WORKS

The model runs on Raspberry Pi, the video feed taken by a Pi camera will be given as the input to the device. The camera will be placed in the center of the top edge of the boundary (ie: laptop camera placement). The device attempts to detect the distance of the eye from the camera and attempts to calculate the deviation angle of the eye pupil from the center position, so the coordinates of the point where the eyes are looking on the screen can be calculated. Once the coordinates of the screen are found it can be used to do the actions of a mouse controller. 

DESIGN DECISIONS

Our first design decision was the primary language in which to implement the system. We choose python because we are familiar with it very much more than the other languages, so we can save time. 

The whole process will be done on the device itself, as we are planning to not use the existing libraries, the focus will be on the accuracy and the efficiency of the model being developed. And may be later the system can include cloud processing.

LANGUAGE

Python - 100%

FILE LIST
README
Eyegaze.py

