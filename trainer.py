import pandas as pd 
import pickle 
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv(r'dataset_3.csv')
print(dataset.head())
X_train = dataset.drop('Label',axis=1)
y_train = dataset['Label']
model =  LogisticRegression()
model.fit(X_train,y_train)
fileName = "classifier3.pkl"
print("Saving Classifier.....")
try:
    with open(fileName,'wb') as file:
        pickle.dump(model,file)
except:
    print("File not found")