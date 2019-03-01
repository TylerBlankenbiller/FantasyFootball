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
    d['HIT'] = x['qb_hit'].sum()
    d['SACK'] = x['sack'].sum()
    return pd.Series(d, index=['COMP', 'ATT', 'YDS', 'LONG', 'TD', 'INT', 'SACK', 'HIT'])

def runs(x):
    d = {}
    d['RATT'] = x['rush_attempt'].sum()
    d['RYDS'] = x['yards_gained'].sum()
    d['RTD'] = x['touchdown'].sum()
    d['FUM'] = x['fumble'].sum()
    d['LST'] = x['fumble_lost'].sum()
    return pd.Series(d, index=['RATT', 'RYDS', 'RTD', 'FUM', 'LST'])
    
def catches(x):
    d = {}
    d['REC'] = x['complete_pass'].sum()
    d['TGTS'] = x['complete_pass'].sum() + x['incomplete_pass'].sum() + x['interception'].sum()
    d['RECYDS'] = x['yards_gained'].sum()
    d['RECTD'] = x['touchdown'].sum()
    return pd.Series(d, index=['REC', 'TGTS', 'RECYDS', 'RECTD'])
    
    
num = 'test'
for idx in range(10):
    print(idx)
    if(idx < 9):
        num = str(idx + 10)
    else:
        num = '0'+str(idx)
    data = pd.read_csv("reg_pbp_20"+num+".csv", low_memory=False)

    players = pd.DataFrame({'Name': ['Nan'], 'Pos': ['Nan'], 'Team': ['Nan'], 'Comp': [0], 'PAtt': [0], 'PYds': [0], 'PLong': [0], 'PTD': [0], 'Int': [0],
                'Sack': [0], 'Rate': [0], 'RYds': [0], 'RTD': [0], 'Fum': [0], 'Lst': [0], 'Rec': [0], 'Tgts': [0], 'RecYds': [0],
                'RecLong': [0], 'RecTD': [0], 'RAtt': [0], 'RYds': [0], 'RLong': [0], 'RTD': [0], 'Fantasy': [0]})
    #players = pd.DataFrame()
    #players.append(temp)

    row2 = 0
    #print(data['touchdown'])
    found = False  

    throw = data.copy()
    throw = throw.loc[(throw['play_type'] == 'pass') | (throw['play_type'] == 'qb_spike')]
    throw.loc[throw.sack == 1, 'yards_gained'] = 0
    throw['name'] = throw['passer_player_name']
    throw = throw.groupby(['passer_player_name', 'game_id', 'posteam', 'name']).apply(throws)

    throw.to_csv('qb_20'+num+'.csv')
    throw = pd.read_csv('qb_20'+num+'.csv', low_memory=False)

    out = throw.to_json(orient='records')[1:-1].replace('},{', '} {')
    with open('qb20'+num+'.txt', 'w') as f:
        f.write(out)


    run = data.copy()
    run = run.loc[run['play_type'] == 'run']
    run['rattempts'] = 1
    run['runYards'] = run['yards_gained']
    run['rAVG'] = run['runYards']/run['rattempts']
    run['rTD'] = run['touchdown']
    run['name'] = run['rusher_player_name']
    run = run.groupby(['rusher_player_name', 'game_id', 'posteam', 'name']).apply(runs)
    #run = run.groupby(['rusher_player_name', 'game_id', 'posteam']).apply(runs)

    run.to_csv('rb_20'+num+'.csv')
    run = pd.read_csv('rb_20'+num+'.csv', low_memory=False)
    out = run.to_json(orient='records')[1:-1].replace('},{', '} {')
    with open('rb20'+num+'.txt', 'w') as f:
        f.write(out)


    #print(run['rattempts'], run['runYards'], run['rAVG'], run['rTD'], run['fumble'], run['fumble_lost'])

    receiver = data.copy()
    receiver = receiver.loc[(receiver['play_type'] == 'pass') | (receiver['play_type'] == 'qb_spike')]
    receiver.loc[receiver.sack == 1, 'yards_gained'] = 0
    receiver['name'] = receiver['receiver_player_name']
    receiver = receiver.groupby(['receiver_player_name', 'name', 'game_id', 'posteam']).apply(catches)

    receiver.to_csv('wr_20'+num+'.csv')
    receiver = pd.read_csv('wr_20'+num+'.csv', low_memory=False)
    out = receiver.to_json(orient='records')[1:-1].replace('},{', '} {')
    with open('wr20'+num+'.txt', 'w') as f:
        f.write(out)



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
    kicker['name'] = kicker['kicker_player_name']
    kicker = kicker.groupby(['kicker_player_name', 'posteam', 'name', 'game_id']).apply(kicks)
    kicker['percent'] = (kicker['short']+kicker['med']+kicker['long'])/kicker['attempt']
    kicker['PAT_percent'] = (kicker['PAT_Made'])/kicker['PAT_Attempt']
    #kicker['fantasy'] = kicker['short']*4 + kicker['med']*5+ kicker['long']*6 + kicker['PAT_Made']*2 - kicker['PAT_Attempt'] - kicker['attempt']
    #- kickers.loc[kickers['field_goal_result'] == 'missed']

    kicker.to_csv('kicker_20'+num+'.csv')
    kicker = pd.read_csv('kicker_20'+num+'.csv', low_memory=False)
    out = kicker.to_json(orient='records')[1:-1].replace('},{', '} {')
    with open('k20'+num+'.txt', 'w') as f:
        f.write(out)


    players = pd.concat([throw, run, receiver, kicker], axis=0, ignore_index=True)
    players = players.groupby(['name', 'game_id', 'posteam']).sum()
    players['year'] = '20'+num
    players['position'] = players['percent'].apply(lambda x: 'K' if x > 0 else 'FB')
    players.loc[(players.RECYDS > 2*players.RYDS) & (players.ATT < 3) & (players.position != 'K'), 'position'] = 'WR'
    players.loc[(players.RECYDS <= 2*players.RYDS) & (players.ATT < 3) & (players.position != 'K'), 'position'] = 'RB'
    players.loc[(players.ATT >= 3) & (players.position != 'K'), 'position'] = 'QB'
    
    #players.to_csv('players_20'+num+'.csv')
    #players = pd.read_csv("players_20"+num+".csv", low_memory=False)
    #players['DATE'] = players['game_id'].apply(lambda x: str(x)[4:8])
    #players['WEEK'] = players.apply(lambda x: 1 if (x['year'] == 208) & (str(x['DATE'][0]) == '9') & (  
    
    
    players.to_csv('players_20'+num+'.csv')
    players = pd.read_csv('players_20'+num+'.csv', low_memory=False)
    out = players.to_json(orient='records')[1:-1].replace('},{', '} {')
    with open('players20'+num+'.txt', 'w') as f:
        f.write(out)