#include players
import pandas as pd


data = pd.read_csv('testLast.csv', low_memory=False)

data = data.loc[data['posteam'] != '0']
df = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular')]

data = data.drop(df.index)

print(len(data))
passes = data.loc[data['fumble_lost'].astype(int) == 1]
print(len(passes))
data = data.drop(passes.index)

data = data.sample(len(passes))

data2 = pd.concat([data, passes])

data2.to_csv('fumbleLost.csv')