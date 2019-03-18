import cv2
import os
import numpy as np
from PIL import Image

path = os.path.join(os.getcwd(),"data")
CATEGORIES = os.listdir(path)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

y_label = []
x_train = []

for category in CATEGORIES:
	file_path =os.path.join(path,category)
	for img in os.listdir(file_path):
		image_path = os.path.join(file_path,img)
		pil_image = Image.open(image_path).convert("L")
		final_image = pil_image.resize((500,500),Image.ANTIALIAS)
		image_array = np.array(pil_image,'uint8')
		print(image_array)
		faces = face_cascade.detectMultiScale(image_array, 1.5, 5)
		for (x,y,w,h) in faces:
			roi = image_array[y:y+h,x:x+w]
			x_train.append(roi)
			y_label.append(CATEGORIES.index(category))
print(y_label)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(x_train,np.array(y_label))
model.save("models/recognize.yml")
