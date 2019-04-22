import pandas as pd


df = pd.read_csv('testLast2013-2017.csv', low_memory=False)

df = df.loc[df['posteam'] != '0']

print(len(df))

to = df.loc[(df['timeout_team'] != '0')]

print(len(to))

df = df.drop(to.index)

df = df.sample(len(to))

data = pd.concat([df, to])

print(len(df))

data.to_csv('timeout.csv')