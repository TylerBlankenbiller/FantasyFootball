import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("allfinal.csv", low_memory=False)
data = data.loc[data.Year == 2018]
data = data.loc[data['posteam'] != '0']
data = data.loc[data['kickoff_attempt'] == 0]
print(data)
    
new = data.Wind.str.split('m',1, expand = True)
data['WSpeed'] = new[0]
data['WDirection'] = new[1]
new = data.Weather.str.split('f',1, expand = True)
data['WTemp'] = new[0]
data['Weather'] = new[1]

#data = data.drop(['Wind'])
data.loc[(data.WTemp =='DOME'), 'WTemp'] = 70
    
    
data['WTemp'].replace('', np.nan, inplace=True)
data['WTemp'].replace(0, np.nan, inplace=True)
data.dropna(subset=['WTemp'], inplace=True) 
data.dropna(subset=['Weather'], inplace=True) 
    
data['duration'] = data['game_seconds_remaining'].diff()
data.loc[data.game_seconds_remaining == 3600, 'duration'] = 0

data['THRpasser_player_id'] = 'THR' + data['passer_player_id']
data['RECreceiver_player_id'] = 'REC' + data['receiver_player_id']
data['RSHrusher_player_id'] = 'RSH' + data['rusher_player_id']
data['KICKkicker_player_id'] = 'KICK' + data['kicker_player_id']
data2 = data[['game_id', 'THRpasser_player_id', 'RECreceiver_player_id', 'RSHrusher_player_id', 'KICKkicker_player_id']].copy()
##headers = data.dtypes.index

data2 = pd.concat([data2, pd.get_dummies(data['THRpasser_player_id'])], axis=1)
data2 = pd.concat([data2, pd.get_dummies(data['RECreceiver_player_id'])], axis=1)
data2 = pd.concat([data2, pd.get_dummies(data['RSHrusher_player_id'])], axis=1)
data2 = pd.concat([data2, pd.get_dummies(data['KICKkicker_player_id'])], axis=1)



##dummy_cols = list(set(headers.columns) - set(headers))

#df = pd.get_dummies(df, columns=dummy_cols)

data2 = data2.groupby(['game_id']).sum()

for c in data2.columns:
    data2[c] = data2[c].apply(lambda x: 1 if x >= 1 else 0)
    
data = pd.merge(data, data2, left_on=['game_id'], right_on=['game_id'], how='left')


data = data.fillna(0)
data = data.drop(columns=['Wind'])
data = data.loc[data['penalty'] == 0]
data = data.drop(columns=['penalty', 'first_down_penalty', 'penalty_team', 'penalty_yards'])
data = data.drop(columns=['kickoff_attempt'])
data.loc[(data.WTemp == '33/51'), 'WTemp'] = '42'
data.loc[(data.WTemp == '53/78'), 'WTemp'] = '65'
data.loc[(data.WTemp == '57/72'), 'WTemp'] = '65'

print(data)
data.to_csv('predict.csv')