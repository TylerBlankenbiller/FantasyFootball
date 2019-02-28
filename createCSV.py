import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def kicks(x):
    d = {}
    d['short'] = x['short'].sum()
    d['med'] = x['med'].sum()
    d['long'] = x['long'].sum()
    d['longest'] = x['kick_distance'].max()
    d['attempt'] = x['attempt'].sum()
    d['PAT_Attempt'] = x['PAT_Attempt'].sum()
    d['PAT_Made'] = x['PAT_Made'].sum() 
    return pd.Series(d, index=['short', 'med', 'long', 'longest', 'attempt', 'PAT_Attempt', 'PAT_Made'])
	
def throws(x):
    d = {}
    d['COMP'] = x['complete_pass'].sum()
    d['ATT'] = x['incomplete_pass'].sum() + x['complete_pass'].sum() + x['interception'].sum()
    d['YDS'] = x['yards_gained'].sum()
    d['Long'] = x['yards_gained'].max()
    d['TD'] = x['touchdown'].sum()
    d['INT'] = x['interception'].sum()
    d['SACK'] = x['sack'].sum()
    d['POS'] = 'QB'
    return pd.Series(d, index=['COMP', 'ATT', 'YDS', 'LONG', 'TD', 'INT', 'SACK', 'POS'])
    


data = pd.read_csv("reg_pbp_2018.csv", low_memory=False)

players = pd.DataFrame({'Name': ['Nan'], 'Pos': ['Nan'], 'Team': ['Nan'], 'Comp': [0], 'PAtt': [0], 'PYds': [0], 'PLong': [0], 'PTD': [0], 'Int': [0],
            'Sack': [0], 'Rate': [0], 'RYds': [0], 'RTD': [0], 'Fum': [0], 'Lst': [0], 'Rec': [0], 'Tgts': [0], 'RecYds': [0],
            'RecLong': [0], 'RecTD': [0], 'RAtt': [0], 'RYds': [0], 'RLong': [0], 'RTD': [0], 'Fantasy': [0]})
#players = pd.DataFrame()
#players.append(temp)

row2 = 0
#print(data['touchdown'])
found = False  

throw = data.copy()
#throw = throw.loc[throw['penalty'] == 0]
throw = throw.loc[(throw['play_type'] == 'pass') | (throw['play_type'] == 'qb_spike')]
throw.loc[throw.sack == 1, 'yards_gained'] = 0
throw = throw.groupby(['passer_player_name', 'game_id', 'posteam']).apply(throws)

throw.to_csv('qb_2018.csv')


run = data
run = run.loc[run['penalty'] == 0]
run = run.loc[run['play_type'] == 'run']
run['name'] = run['rusher_player_name']
run['rattempts'] = 1
#run = run.groupby(['rusher_player_name', 'posteam']).sum()
#run = run.sort_values(['yards_gained'])
run['runYards'] = run['yards_gained']
run['rAVG'] = run['runYards']/run['rattempts']
run['rTD'] = run['touchdown']
#print(run['rattempts'], run['runYards'], run['rAVG'], run['rTD'], run['fumble'], run['fumble_lost'])

receiver = data
receiver = receiver.loc[receiver['penalty'] == 0]
receiver = receiver.loc[receiver['play_type'] == 'pass']
receiver['name'] = receiver['receiver_player_name'] 
receiver['target'] = 1
#receiver = receiver.groupby(['receiver_player_name', 'posteam']).sum()
#receiver = receiver.sort_values(['yards_gained'])
#print(receiver)


kicker = data
kicker = kicker.loc[kicker['penalty'] == 0]
kicker = kicker.loc[(kicker['play_type'] == 'field_goal') | (kicker.play_type == 'extra_point')]
kicker['attempt'] = 0
kicker['short'] = 0
kicker['med'] = 0
kicker['long'] = 0
kicker['PAT_Made'] = 0
kicker['PAT_Attempt'] = 0
kicker.loc[kicker.play_type == 'field_goal', 'attempt'] += 1
kicker.loc[kicker.extra_point_attempt == 1, 'PAT_Attempt'] += 1
kicker.loc[kicker.extra_point_result == 'good', 'PAT_Made'] += 1
kicker.loc[(kicker.kick_distance <= 39) & (kicker.field_goal_result == 'made'), 'short'] += 1
kicker.loc[(kicker.kick_distance > 39) & (kicker.kick_distance <= 49) & (kicker.field_goal_result == 'made'), 'med'] += 1
kicker.loc[(kicker.kick_distance >= 50) & (kicker.field_goal_result == 'made'), 'long'] += 1
kicker = kicker.groupby(['kicker_player_name', 'posteam']).apply(kicks)
kicker['percent'] = (kicker['short']+kicker['med']+kicker['long'])/kicker['attempt']
kicker['PAT_percent'] = (kicker['PAT_Made'])/kicker['PAT_Attempt']
#kicker['fantasy'] = kicker['short']*4 + kicker['med']*5+ kicker['long']*6 + kicker['PAT_Made']*2 - kicker['PAT_Attempt'] - kicker['attempt']
#- kickers.loc[kickers['field_goal_result'] == 'missed']

kicker.to_csv('kicker_2018.csv')