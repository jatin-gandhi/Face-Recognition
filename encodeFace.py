import glob
import face_recognition
import cv2
import numpy as np
import pandas as pd

class EncodeFaces:
    def __init__(self):
        self.knownEncodings=[]
    def extractImages(self):
        self.images = glob.glob("*.jpeg")
        print(len(self.images))
    def detectAndEncode(self):
        self.extractImages()
        for image in self.images:
            img = cv2.imread(image)
            self.locations = face_recognition.face_locations(img)
            if self.locations:
                self.encodings = face_recognition.face_encodings(img,self.locations)
                self.knownEncodings.append(list(self.encodings[0]))
                print(" DONE ")
            else:
                print("NOT DONE")
    def createDataset(self):
        self.detectAndEncode()
        self.knownEncodings = np.array(self.knownEncodings)
        print(self.knownEncodings.shape)
        self.dataframe = pd.DataFrame(self.knownEncodings)
        print(self.dataframe.head())
        self.dataframe.to_csv("dataset4.csv")

obj = EncodeFaces()
obj.createDataset()
