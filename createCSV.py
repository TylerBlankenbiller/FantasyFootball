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
    week = {    1:  ['20180906', '20180909', '20180910',#2018
                    '20170907', '20170910', '20170911',#2017
					'20160908', '20160911', '20160912',#2016
					'20150910', '20150913', '20150914',#2015
					'20140904', '20140907', '20140908',#2014
					'20130905', '20130908', '20130909',#2013
                    ,
                2:  ['20180913', '20180916', '20180917',#2018
                    '20170914', '20170917', '20170918',#2017
					'20160915', '20160918', '20160919',#2016
					'20150917', '20150920', '20150921',#2015
					'20140911', '20140914', '20140915',#2014
					'20130912', '20130915', '20130916',#2013
                    , 
                3:  ['20180920', '20180923', '20180924',#2018
                    '20170921', '20170924', '20170925',#2017
					'20160922', '20160925', '20160926',#2016
					'20150924', '20150927', '20150928',#2015
					'20140918', '20140921', '20140922',#2014
					'20130919', '20130922', '20130923',#2013
                    ,
                4:  ['20180927', '20180930', '20181001',#2018
                    '20170928', '20171001', '20171002',#2017
					'20160929', '20161002', '20161003',#2016
					'20151001', '20151004', '20151005',#2015
					'20140925', '20140928', '20140929',#2014
					'20130926', '20130929', '20130930',#2013
                    ,
                5:  ['20181004', '20181007', '20181008',#2018
                    '20171005', '20171008', '20171009',#2017
					'20161006', '20161009', '20161010',#2016
					'20151008', '20151011', '20151012',#2015
					'20141002', '20141005', '20141006',#2014
					'20131003', '20131006', '20131007',#2013
                    ,
                6:  ['20181011', '20181014', '20181015',#2018
					'20171012', '20171015', '20171016',#2017
					'20161013', '20161016', '20161017',#2016
					'20151015', '20151018', '20151019',#2015
					'20141009', '20141012', '20141013',#2014
					'20131010', '20131013', '20131014',#2013
                    ,
                7:  ['20181018', '20181021', '20181022',#2018
					'20171019', '20171022', '20171023',#2017
					'20161020', '20161023', '20161024',#2016
					'20151022', '20151025', '20151026',#2015
					'20141016', '20141019', '20141020',#2014
					'20131017', '20131020', '20131021',#2013
                    ,
                8:  ['20181025', '20181028', '20181029',#2018
					'20171026', '20171029', '20171030',#2017
					'20161027', '20161030', '20161031',#2016
					'20151029', '20151101', '20151102',#2015
					'20141023', '20141026', '20141027',#2014
					'20131024', '20131027', '20131028',#2013
                    ,
                9:  ['20181101', '20181104', '20181105',#2018
					'20171102', '20171105', '20171106',#2017
					'20161103', '20161106', '20161107',#2016
					'20151105', '20151108', '20151109',#2015
					'20141030', '20141102', '20141103',#2014
					'20131031', '20131103', '20131104',#2013
                    ,
                10:  ['20181108', '20181111', '20181112',#2018
					 '20171109', '20171112', '20171113',#2017
					 '20161110', '20161113', '20161114',#2016
					 '20151112', '20151115', '20151116',#2015
					 '20141106', '20141109', '20141110',#2014
					 '20131107', '20131110', '20131111',#2013
                    ,
                11:  ['20181115', '20181118', '20181119',#2018
					 '20171116', '20171119', '20171120',#2017
					 '20161117', '20161120', '20161121',#2016
					 '20151119', '20151122', '20151123',#2015
					 '20141113', '20141116', '20141117',#2014
					 '20131114', '20131117', '20131118',#2013
                    ,
                12:  ['20181122', '20181125', '20181126',#2018
					 '20171123', '20171126', '20171127',#2017
					 '20161124', '20161127', '20161128',#2016
					 '20151126', '20151129', '20151130',#2015
					 '20141120', '20141123', '20141124',#2014
					 '20131121', '20131124', '20131125',#2013
                    ,
                13:  ['20181129', '20181202', '20181203',#2018
					 '20171130', '20171203', '20171204',#2017
					 '20161201', '20161204', '20161205',#2016
					 '20151203', '20151206', '20151207',#2015
					 '20141127', '20141130', '20141201',#2014
					 '20131128', '20131201', '20131202',#2013
                    ,
                14:  ['20181206', '20181209', '20181210',#2018
					 '20171207', '20171210', '20171211',#2017
					 '20161208', '20161211', '20161212',#2016
					 '20151210', '20151213', '20151214',#2015
					 '20141204', '20141207', '20141208',#2014
					 '20131205', '20131208', '20131209',#2013
                    ,
                15:  ['20181213', '20181215', '20181216', '20181217',#2018
					 '20171214', '20171216', '20171217', '20171218',#2017
					 '20161215', '20161217', '20161218', '20161219',#2016
					 '20151217', '20151219', '20151220', '20151221',#2015
					 '20141211', '20141213', '20141214', '20141215',#2014
					 '20131212', '20131214', '20131215', '20131216',#2013
                    ,
                16:  ['20181222', '20181223', '20181224',#2018
					 '20171223', '20171224', '20171225',#2017
					 '20161222', '20161224', '20161225', '20161226',#2016
					 '20151224', '20151226', '20151227', '20151228',#2015
					 '20141218', '20141220', '20141221', '20141222',#2014
                    ,
                17:  ['20181230',#2018
					 '20181231',#2017
					 '20170101',#2016
					 '20160103',#2015
					 '20141228',#2014
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