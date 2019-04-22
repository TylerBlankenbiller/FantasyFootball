import pandas as pd


df = pd.read_csv('testLast.csv', low_memory=False)

print(len(df))

df = df.loc[df['Year'] != 2018]

print(len(df))

df.to_csv('timeout.csv')