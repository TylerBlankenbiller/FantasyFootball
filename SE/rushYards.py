#include players
import pandas as pd


data = pd.read_csv('testLast.csv', low_memory=False)

data = data.loc[data['posteam'] != '0']
df = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular')]

data = data.drop(df.index)

print(len(data))

runs = data.loc[data['rush_attempt'].astype(int) == 1]

data = data.drop(runs.index)

data = data.sample(len(runs))

data2 = pd.concat([data, runs])

data2.to_csv('rushYards.csv')
