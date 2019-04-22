import pandas as pd


df = pd.read_csv('testLast.csv', low_memory=False)

df = df.loc[df['posteam'] != '0']

print(len(df))

to = df.loc[(df['run_location'] != '0')]

print(len(to))

df = df.drop(to.index)

if len(to) > 10000:
    to = to.sample(10000)

df = df.sample(len(to))

data = pd.concat([df, to])

print(len(df))

data.to_csv('runLocation.csv')