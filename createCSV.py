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
    
def checkWeek(x):
    week = {    1:  ['20180906', '20180909', '20180910']#2018
                    ['20170907', '20170910', '20170911']#2017
                    ,
                2:  ['20180913', '20180916', '20180917']#2018
                    ['20170914', '20170917', '20170918']#2017
                    , 
                3:  ['20180920', '20180923', '20180924']#2018
                    ['20170921', '20170924', '20170925']#2017
                    ,
                4:  ['20180927', '20180930', '20181001']#2018
                    ['20170928', '20171001', '20171002']#2017
                    ,
                5:  ['20181004', '20181007', '20181008']#2018
                    ['20171005', '20171008', '20171009']#2017
                    ,
                6:  ['20181011', '20181014', '20181015']#2018
                    ,
                7:  ['20181018', '20181021', '20181022']#2018
                    ,
                8:  ['20181025', '20181028', '20181029']#2018
                    ,
                9:  ['20181101', '20181104', '20181105']#2018
                    ,
                10:  ['20181108', '20181111', '20181112']#2018
                    ,
                11:  ['20181115', '20181118', '20181119']#2018
                    ,
                12:  ['20181122', '20181125', '20181126']#2018
                    ,
                13:  ['20181129', '20181202', '20181203']#2018
                    ,
                14:  ['20181206', '20181209', '20181210']#2018
                    ,
                15:  ['20181213', '20181215', '20181216', '20181217']#2018
                    ,
                16:  ['20181222', '20181223', '20181224']#2018
                    ,
                17:  ['20181230']#2018
                }
    for key, value in week.items():#Check Dictionary Keys
        for i in range(len(value)):
            if str(x) == value[i]:
                return key
    return 0

num = 'test'
for idx in range(10):
    idx = 8
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
    throw = throw.groupby(['game_id', 'passer_player_name', 'game_id', 'posteam', 'defteam', 'name']).apply(throws)

    throw.to_csv('qb_20'+num+'.csv')
    throw = pd.read_csv('qb_20'+num+'.csv', low_memory=False)

    #out = throw.to_json(orient='records')[1:-1].replace('},{', '} {')
    #with open('qb20'+num+'.txt', 'w') as f:
    #    f.write(out)


    run = data.copy()
    run = run.loc[run['play_type'] == 'run']
    run['rattempts'] = 1
    run['runYards'] = run['yards_gained']
    run['rAVG'] = run['runYards']/run['rattempts']
    run['rTD'] = run['touchdown']
    run['name'] = run['rusher_player_name']
    run = run.groupby(['game_id', 'rusher_player_name', 'game_id', 'posteam', 'defteam', 'name']).apply(runs)

    run.to_csv('rb_20'+num+'.csv')
    run = pd.read_csv('rb_20'+num+'.csv', low_memory=False)
    #out = run.to_json(orient='records')[1:-1].replace('},{', '} {')
    #with open('rb20'+num+'.txt', 'w') as f:
    #    f.write(out)


    #print(run['rattempts'], run['runYards'], run['rAVG'], run['rTD'], run['fumble'], run['fumble_lost'])

    receiver = data.copy()
    receiver = receiver.loc[(receiver['play_type'] == 'pass') | (receiver['play_type'] == 'qb_spike')]
    receiver.loc[receiver.sack == 1, 'yards_gained'] = 0
    receiver['name'] = receiver['receiver_player_name']
    receiver = receiver.groupby(['game_id', 'receiver_player_name', 'name', 'game_id', 'posteam', 'defteam']).apply(catches)

    receiver.to_csv('wr_20'+num+'.csv')
    receiver = pd.read_csv('wr_20'+num+'.csv', low_memory=False)
    #out = receiver.to_json(orient='records')[1:-1].replace('},{', '} {')
    #with open('wr20'+num+'.txt', 'w') as f:
    #    f.write(out)



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
    kicker = kicker.groupby(['game_id', 'kicker_player_name', 'posteam', 'name', 'game_id', 'defteam']).apply(kicks)
    kicker['percent'] = (kicker['short']+kicker['med']+kicker['long'])/kicker['attempt']
    kicker['PAT_percent'] = (kicker['PAT_Made'])/kicker['PAT_Attempt']

    kicker.to_csv('kicker_20'+num+'.csv')
    kicker = pd.read_csv('kicker_20'+num+'.csv', low_memory=False)
    #out = kicker.to_json(orient='records')[1:-1].replace('},{', '} {')
    #with open('k20'+num+'.txt', 'w') as f:
    #    f.write(out)


    players = pd.concat([throw, run, receiver, kicker], axis=0, ignore_index=True)
    players['date'] = players['game_id'].astype(str).str[:-2].astype(np.int64)
    players['week'] = np.vectorize(checkWeek)(players['date'])
    players = players.groupby(['week', 'date', 'game_id', 'name', 'game_id', 'posteam', 'defteam']).sum()
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
    break