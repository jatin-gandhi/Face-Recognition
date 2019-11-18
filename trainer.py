import pandas as pd 
import pickle 
from sklearn.linear_model import LogisticRegression
class Trainer:
    def __init__(self):
        pass
    def Dataset(self):
        self.dataset = pd.read_csv(r'dataset.csv')
        print(self.dataset.head())
    def trainModel(self):
        self.Dataset()
        self.X_train = self.dataset.drop('Label',axis=1)
        self.y_train = self.dataset['Label']
        self.model =  LogisticRegression()
        self.model.fit(self.X_train,self.y_train)
        self.fileName = "classifier.pkl"
        print("Saving Classifier.....")
        try:
            with open(self.fileName,'wb') as file:
                pickle.dump(self.model,file)
        except:
            print("File not found")

obj = Trainer()
obj.trainModel()
