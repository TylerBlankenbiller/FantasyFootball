#include players
import pandas as pd


data = pd.read_csv('testLast.csv', low_memory=False)

data = data.loc[data['posteam'] != '0']
df = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular')]

data = data.drop(df.index)

print(len(data))
passes = data.loc[data['extra_point_attempt'].astype(int) == 1].copy()
passes.loc[(passes.extra_point_result == 'good'), 'extra_point_result'] = '1'
passes.loc[(passes.extra_point_result == 'failed'), 'extra_point_result'] = '0'
passes.loc[(passes.extra_point_result == 'blocked'), 'extra_point_result'] = '0'
passes = passes.loc[passes.extra_point_result != 'aborted']

miss = passes.loc[(passes['extra_point_result'] == '0')]
print(len(passes))
print(len(miss))

passes =  passes.sample(len(miss))

data2 = pd.concat([passes, miss])
data2['extra_point_result'] = data2['extra_point_result'].astype(int)

data2.to_csv('extraPoint.csv')
