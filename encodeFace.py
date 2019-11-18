import glob
import face_recognition
import cv2
import numpy as np
import pandas as pd
images = glob.glob("*.jpeg")
knownEncodings=[]
print(len(images))
for image in images:
    img = cv2.imread(image)
    locations = face_recognition.face_locations(img)
    if locations:
        encodings = face_recognition.face_encodings(img,locations)
        knownEncodings.append(list(encodings[0]))
        print(" DONE ")
    else:
        print("NOT DONE")

knownEncodings = np.array(knownEncodings)
print(knownEncodings.shape)
df = pd.DataFrame(knownEncodings)
print(df.head())
df.to_csv("dataset.csv")