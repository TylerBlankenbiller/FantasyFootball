import pandas as pd

df09 = pd.read_csv('players_2009.csv', low_memory=False)
df10 = pd.read_csv('players_2010.csv', low_memory=False)
df11 = pd.read_csv('players_2011.csv', low_memory=False)
df12 = pd.read_csv('players_2012.csv', low_memory=False)
df13 = pd.read_csv('players_2013.csv', low_memory=False)
df14 = pd.read_csv('players_2014.csv', low_memory=False)
df15 = pd.read_csv('players_2015.csv', low_memory=False)
df16 = pd.read_csv('players_2016.csv', low_memory=False)
df17 = pd.read_csv('players_2017.csv', low_memory=False)
df18 = pd.read_csv('players_2018.csv', low_memory=False)

data = pd.concat([df09, df10, df11, df12, df13, df14, df15, df16, df17, df18])

data.to_csv('allActualStats.csv')