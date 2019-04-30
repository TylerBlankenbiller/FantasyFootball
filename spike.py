#include players
import pandas as pd


data = pd.read_csv('testLast.csv', low_memory=False)

df = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular')]

data = data.drop(df.index)

knee = data.loc[data['qb_spike'] == 1]

data = data.sample(len(knee))

data2 = pd.concat([data, knee])

print(len(data))

data2.to_csv('spike.csv')
