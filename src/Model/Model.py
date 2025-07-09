from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from tensorflow.python.ops.gen_resource_variable_ops import resource_scatter_update


class Model:
    def __init__(self,kernel='linear'):
        self.model = SVC(kernel=kernel)
        self.scaler = StandardScaler()

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.x_train_scaled = None
        self.x_test_scaled = None

        self.y_pred = None

    def preprocessing(self, df):
        x = df[['mar', 'lear', 'rear']]
        y = df['drowsy']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, test_size=.2, random_state=10)
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.scaler.transform(self.x_test)


    def train(self):
        self.model.fit(self.x_train_scaled, self.y_train)


    def predict(self,):
        self.y_pred = self.model.predict(self.x_test_scaled)


    def evaluation(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        cm = confusion_matrix(self.y_test, self.y_pred)
        cr = classification_report(self.y_test, self.y_pred)
        return accuracy, cm, cr

    def predict_one(self, mar, lear, rear):
        input_array = np.array([[mar, lear, rear]])
        input_scaled = self.scaler.transform(input_array)
        result = self.model.predict(input_scaled)
        return result


if __name__ == '__main__':

    model = Model()

    data = "../../resources/data/output/top_20.csv"
    df = pd.read_csv(data)

    model.preprocessing(df)
    model.train()
    model.predict()

    acc, cm, cr = model.evaluation()

    print(acc)
    print(cm)
    print(cr)









