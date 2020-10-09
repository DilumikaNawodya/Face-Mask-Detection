# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:27:26 2020

@author: Dilumika
"""

from keras.models import load_model
import cv2
import numpy as np
import os


model = load_model('model-010.model')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels = {0:'MASK',1:'WITHOUT MASK'}
colors = {0:(0,255,0),1:(255,0,0)}

DATADIR = 'E:\Education\Language\Projects\Face Mask Detection\Face Mask Dataset\Predictions'
CATEGORY = ["To be","Done"]

opt = int(input("Enter 1 or 0 : "))
i = 1
if opt==1:
    open_path = os.path.join(DATADIR,CATEGORY[0])
    save_dir = os.path.join(DATADIR,CATEGORY[1])
    for img in os.listdir(open_path):
        
        frame = cv2.imread(os.path.join(open_path,img))
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        
        for (x,y,w,h) in faces:
            
            face = gray[y:y+h,x:x+w]
            resized_image = cv2.resize(face,(50,50))
            normalized_image = resized_image/255.0
            
            reshaped_image = np.reshape(normalized_image,(1,50,50,1))
            
            result = model.predict(reshaped_image)
            
            label = np.argmax(result,axis=1)[0]
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),colors[label],2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),colors[label],-1)
            
            cv2.putText(frame,labels[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        cv2.imwrite(os.path.join(save_dir, str(i)+".jpg"), frame)
        i += 1


else:
    video_capture = cv2.VideoCapture(0)
    
    while True:
        _,frame = video_capture.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        
        for (x,y,w,h) in faces:
            
            face = gray[y:y+h,x:x+w]
            resized_image = cv2.resize(face,(50,50))
            normalized_image = resized_image/255.0
            
            reshaped_image = np.reshape(normalized_image,(1,50,50,1))
            
            result = model.predict(reshaped_image)
            
            label = np.argmax(result,axis=1)[0]

            cv2.rectangle(frame,(x,y),(x+w,y+h),colors[label],2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),colors[label],-1)
            
            cv2.putText(frame,labels[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
        cv2.imshow('Video',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()