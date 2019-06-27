#include players
import pandas as pd


data = pd.read_csv('testLast.csv', low_memory=False)

data = data.loc[data['posteam'] != '0']
df = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular')]

data = data.drop(df.index)

print(len(data))
passes = data.loc[data['field_goal_attempt'].astype(int) == 1].copy()
passes.loc[(passes.field_goal_result == 'made'), 'field_goal_result'] = '1'
passes.loc[(passes.field_goal_result == 'missed'), 'field_goal_result'] = '0'
passes.loc[(passes.field_goal_result == 'blocked'), 'field_goal_result'] = '0'

miss = passes.loc[(passes['field_goal_result'] == '0')]
print(len(passes))
print(len(miss))

passes =  passes.sample(len(miss))

data2 = pd.concat([passes, miss])
data2['field_goal_result'] = data2['field_goal_result'].astype(int)

data2.to_csv('fieldGoal.csv')
