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
balance_data = balance_data.sample(frac=0.5, replace=True, random_state=1)
balance_data = balance_data.fillna(0)
print("Dataset Lenght:: ", len(balance_data))
print("Dataset Shape:: ", balance_data.shape)

print("Dataset:: ", balance_data.head())

#Predict
X = balance_data.values[:, 0:40]
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
temp = onehot_encoder.fit_transform(X[:, 30:31].astype(str))
Predict= np.append(Predict, temp, 1)
temp = X[:, 31:32]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(X[:, 32:33].astype(str))
Predict= np.append(Predict, temp, 1)
temp = X[:, 33:34]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(X[:, 34:40].astype(str))
Predict= np.append(Predict, temp, 1)
X = Predict
print(Predict)

#Outcome
temp = balance_data.values[:, 124:127]#Good
Predict = temp
temp = onehot_encoder.fit_transform(balance_data.values[:, 122:124].astype(str))#Good
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 121:122]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 119:121].astype(str))
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 118:119]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 115:118].astype(str))#run location is first
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 113:115]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 112:113].astype(str))
Predict= np.append(Predict, temp, 1)
print("HERE1")
temp = balance_data.values[:, 105:112]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 104:105].astype(str))#play type
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 101:104]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 99:101].astype(str))#replay
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 97:99]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 96:97].astype(str))#penalty team
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 95:96]
Predict= np.append(Predict, temp, 1)
temp = onehot_encoder.fit_transform(balance_data.values[:, 91:95].astype(str))#player ID
Predict= np.append(Predict, temp, 1)
temp = balance_data.values[:, 40:91]
Predict= np.append(Predict, temp, 1)
Y = Predict
del Predict
del temp
del balance_data

print("HERE")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
del X
del Y

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
