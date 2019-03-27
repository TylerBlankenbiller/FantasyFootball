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
X = balance_data.values[:, 0:42]
temp = onehot_encoder.fit_transform(X[:, 0:6].astype(str))#+X[:, 6:9]+onehot_encoder.fit_transform(X[:, 10].astype(str))+X[:, 11:29]+onehot_encoder.fit_transform(X[:, 30:32].astype(str))+X[:, 33]+onehot_encoder.fit_transform(X[:, 34].astype(str))+X[:, 35]+onehot_encoder.fit_transform(X[:, 36:41].astype(str))
Predict = temp
print(Predict)
temp = X[:, 6:10]
Predict= np.append(Predict, temp, 1)
print(Predict)
temp = onehot_encoder.fit_transform(X[:, 10:11].astype(str))
#temp = temp.reshape(-1, 1)
print(Predict)
Predict= np.append(Predict, temp, 1)
temp = X[:, 11:30]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(X[:, 30:33].astype(str))
Predict= np.append(Predict, temp, 1)
temp = X[:, 33:34]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(X[:, 34:35].astype(str))
Predict= np.append(Predict, temp, 1)
temp = X[:, 35:36]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(X[:, 36:42].astype(str))
Predict= np.append(Predict, temp, 1)
X = Predict
print(Predict)

#Outcome
temp = balance_data.values[:, 126:129]
Predict = temp
temp = onehot_encoder.fit_transform(balance_data.values[:, 124:126].astype(str))
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 123:124]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 121:123].astype(str))
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 120:121]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 117:120].astype(str))#run location is first
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 115:117]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 114:115].astype(str))
Predict= np.append(Predict, temp, 1)
print("HERE1")
temp = balance_data.values[:, 107:117]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 106:107].astype(str))#play type
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 103:106]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 101:103].astype(str))#replay
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 99:101]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 98:99].astype(str))#penalty team
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 97:98]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 93:97].astype(str))#player ID
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 42:93]
Predict= np.append(Predict, temp, 1)
Y = Predict
del Predict
del temp
del balance_data

print("HERE")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
