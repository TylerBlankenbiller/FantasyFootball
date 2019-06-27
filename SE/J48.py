import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

y2009 = pd.read_csv("reg_pbp_2009.csv", low_memory=False)
y2010 = pd.read_csv("reg_pbp_2010.csv", low_memory=False)
y2011 = pd.read_csv("reg_pbp_2011.csv", low_memory=False)
y2012 = pd.read_csv("reg_pbp_2012.csv", low_memory=False)
y2013 = pd.read_csv("reg_pbp_2013.csv", low_memory=False)
y2014 = pd.read_csv("reg_pbp_2014.csv", low_memory=False)
y2015 = pd.read_csv("reg_pbp_2015.csv", low_memory=False)
y2016 = pd.read_csv("reg_pbp_2016.csv", low_memory=False)
y2017 = pd.read_csv("reg_pbp_2017.csv", low_memory=False)
y2018 = pd.read_csv("reg_pbp_2018.csv", low_memory=False)

balance_data = pd.concat([y2009, y2010, y2011, y2012, y2013, y2014, y2015, y2016, y2017], axis=0, ignore_index=True)

print ("Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)

print("Dataset:: ")
balance_data.head()


