import cv2
import numpy as np
import face_recognition

img = face_recognition.load_image_file('dataset/frost.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('dataset/frost2.jpeg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(img)[0]
encodeImg = face_recognition.face_encodings(img)[0]
cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeImgTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


result = face_recognition.compare_faces([encodeImg],encodeImgTest)
faceDis = face_recognition.face_distance([encodeImg],encodeImgTest)
print(result,faceDis)


cv2.putText(imgTest,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('image1',img)
cv2.imshow('image4 test',imgTest)
cv2.waitKey(0)