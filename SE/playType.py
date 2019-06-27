import pandas as pd


data = pd.read_csv('testLast.csv', low_memory=False)

data = data.loc[data['posteam'] != '0']
data = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular')]

print(len(data))

print(len(data.loc[data['punt_attempt'].astype(int) == 1]))
print(len(data.loc[data['pass_attempt'].astype(int) == 1]))
print(len(data.loc[data['rush_attempt'].astype(int) == 1]))
print(len(data.loc[data['field_goal_attempt'].astype(int) == 1]))

punts = data.loc[data['punt_attempt'].astype(int) == 1]
throws = data.loc[data['pass_attempt'].astype(int) == 1]
runs = data.loc[data['rush_attempt'].astype(int) == 1]
goals = data.loc[data['field_goal_attempt'].astype(int) == 1]

throws = throws.sample(len(goals))
runs = runs.sample(len(goals))
punts = punts.sample(len(goals))
print(len(throws))
print(len(runs))
print(len(punts))
print(len(goals))

data2 = pd.concat([throws, runs, goals, punts])

data2.to_csv('playType.csv')
