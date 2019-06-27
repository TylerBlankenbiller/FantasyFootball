#include players
import pandas as pd


data = pd.read_csv('testLast.csv', low_memory=False)

data = data.loc[data['posteam'] != '0']
df = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular')]

data = data.drop(df.index)

print(len(data))
data = data.loc[data['extra_point_attempt'].astype(int) == 0]
data = data.loc[data['two_point_attempt'].astype(int) == 0]

data.to_csv('duration.csv')
