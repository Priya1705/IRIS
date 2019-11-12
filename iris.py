from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

dataset= pd.read_csv("iris.data")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# encoding values of y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
# print(X)
# print(y)

dataset=pd.DataFrame(dataset);
X=pd.DataFrame(X);
y=pd.DataFrame(y);
y=y.values.ravel()

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
x_new = np.array([[4.9,3.0,1.4,0.2]])
predicted=model.predict(x_new)
if(predicted==0):
	print("setosa")
elif(predicted==1):
	print("versicolor")
else:
	print("virginica")
