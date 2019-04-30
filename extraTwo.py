#include players
import pandas as pd


data = pd.read_csv('testLast.csv', low_memory=False)

data = data.loc[data['posteam'] != '0']
df = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular')]

data = data.drop(df.index)

print(len(data))
passes = data.loc[data['extra_point_attempt'].astype(int) == 1]
print(len(passes))
twos = data.loc[data['two_point_attempt'].astype(int) == 1]
print(len(twos))

passes = twos.sample(len(twos))

data2 = pd.concat([passes, twos])

data2.to_csv('extraTwo.csv')
