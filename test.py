import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./1500.jpg',cv2.IMREAD_COLOR)

newface_cascade = cv2.CascadeClassifier('./cascadeH5.xml')
# face_cascade = cv2.CascadeClassifier('C:/mini-projects/facial/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
# upper_cascade = cv2.CascadeClassifier('C:/mini-projects/facial/opencv/data/haarcascades/haarcascade_upperbody.xml')
# low_cascade = cv2.CascadeClassifier('C:/mini-projects/facial/opencv/data/haarcascades/haarcascade_lowerbody.xml')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )

newfaces = newface_cascade.detectMultiScale(gray, 1.1, 4)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# low = low_cascade.detectMultiScale(gray, 1.1, 3)
# upper = upper_cascade.detectMultiScale(gray, 1.1, 1)
newFacesArray = []
for x in range(1,50):
	newFacesArray.append(newface_cascade.detectMultiScale(gray, 1.0+x*0.1, 3,0,(0,0),(100,100)))
	pass

for newfaces in newFacesArray:
	i=0
	i += 1
	for (x,y,w,h) in newfaces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0+(i*10),0+(i*10),0+(i*10),25))
	print(i)

# for (x,y,w,h) in faces:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (255,122,122),2)
# for (x,y,w,h) in low:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0),2)
# for (x,y,w,h) in upper:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0),2)
    
cv2.imshow('image',img)
cv2.waitKey(0) # If you don'tput this line,thenthe image windowis just a flash. If you put any number other than 0, the same happens.
cv2.destroyAllWindows()
 