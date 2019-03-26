import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#model selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

onehot_encoder = OneHotEncoder(sparse=False)
label_encoder = LabelEncoder()

balance_data = pd.read_csv("given.csv", low_memory=False)
balance_data = balance_data.fillna(0)
print("Dataset Lenght:: ", len(balance_data))
print("Dataset Shape:: ", balance_data.shape)

print("Dataset:: ", balance_data.head())

#Predict
X = balance_data.values[:, 0:41]
temp = onehot_encoder.fit_transform(X[:, 0:5].astype(str))#+X[:, 6:9]+onehot_encoder.fit_transform(X[:, 10].astype(str))+X[:, 11:29]+onehot_encoder.fit_transform(X[:, 30:32].astype(str))+X[:, 33]+onehot_encoder.fit_transform(X[:, 34].astype(str))+X[:, 35]+onehot_encoder.fit_transform(X[:, 36:41].astype(str))
Predict = temp
temp = X[:, 6:9]
Predict= np.concatenate((Predict, temp))

print(X)

#Outcome
Y = balance_data.values[:, 42:129]
Y = onehot_encoder.fit_transform(Y.astype(str))
print("HERE")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
