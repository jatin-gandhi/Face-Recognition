import pickle
import cv2
import face_recognition
import numpy as np
from threading import Timer
class Predictor:
    def __init__(self):
        pass

    def Model(self):
        try:
            with open("classifier.pkl","rb") as file:
                self.model = pickle.load(file)
        except:
            print("File not found")

    def IdentifyFace(self):
        self.Model()
        cap = cv2.VideoCapture(0)
        while True:
            ret,frame = cap.read() 
            if ret:
                locations = face_recognition.face_locations(frame)
                if locations:
                    top, right, bottom, left = locations[0]
                    frame = cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0),10)
                    encodings = face_recognition.face_encodings(frame,locations)
                    test = encodings[0].reshape(1,-1)
                    predict = self.model.predict(test)
                    print(predict)
                    cv2.imshow('fame',frame)
                    if cv2.waitKey(1) & 0xFF==27:
                        break
        cap.release()
        cv2.destroyAllWindows()



obj = Predictor()
timer = Timer(5.0,obj.IdentifyFace)
timer.start()
print("Timer started")


