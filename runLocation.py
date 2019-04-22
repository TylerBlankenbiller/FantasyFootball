import pandas as pd


df = pd.read_csv('testLast2013-2017.csv', low_memory=False)

df = df.loc[df['posteam'] != '0']

print(len(df))

to = df.loc[(df['run_location'] != '0')]

print(len(to))

df = df.drop(to.index)

df = df.sample(20000)

data = pd.concat([df, to])

print(len(df))

data.to_csv('runLocation.csv')