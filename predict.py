import pickle
import cv2
import face_recognition
import numpy as np

try:
    with open("classifier3.pkl","rb") as file:
        model = pickle.load(file)
except:
    print("File not found")
    
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()   
    locations = face_recognition.face_locations(frame)
    if locations:
        top, right, bottom, left = locations[0]
        frame = cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0),10)
        encodings = face_recognition.face_encodings(frame,locations)
        test = encodings[0].reshape(1,-1)
        predict = model.predict(test)
        print(predict)
    cv2.imshow('fame',frame)
    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()