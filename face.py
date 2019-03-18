import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')


model = cv2.face.LBPHFaceRecognizer_create()
model.read("models/recognize.yml")

path = os.path.join(os.getcwd(),"data")
CATEGORIES = os.listdir(path)

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255,0,0)
stroke = 2

cap = cv2.VideoCapture(0)

while True:
    try:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,stroke)
            roi_gray = gray[y:y+h,x:x+w]
            id_,conf =model.predict(roi_gray)
            print(conf,CATEGORIES[id_])
            

            cv2.putText(img,CATEGORIES[id_],(x,y),font,1,color,stroke,cv2.LINE_AA)
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()