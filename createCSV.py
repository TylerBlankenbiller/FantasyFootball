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
	
def ko(x):
    d = {}
    d['KICKOFFTD'] = x['touchdown'].sum()
    return pd.Series(d, index=['KICKOFFTD'])
    
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
					'20120905', '20120909', '20120910',#2012
					'20110908', '20110911', '20110912',#2011
					'20100909', '20100912', '20100913',#2010
					'20090910', '20090913', '20090914']#2009
                    ,
                2:  ['20180913', '20180916', '20180917',#2018
                    '20170914', '20170917', '20170918',#2017
					'20160915', '20160918', '20160919',#2016
					'20150917', '20150920', '20150921',#2015
					'20140911', '20140914', '20140915',#2014
					'20130912', '20130915', '20130916',#2013
					'20120913', '20120916', '20120917',#2012
					'20110915', '20110918', '20110919',#2011
					'20100919', '20100920',#2010
					'20090920', '20090921']#2009
                    , 
                3:  ['20180920', '20180923', '20180924',#2018
                    '20170921', '20170924', '20170925',#2017
					'20160922', '20160925', '20160926',#2016
					'20150924', '20150927', '20150928',#2015
					'20140918', '20140921', '20140922',#2014
					'20130919', '20130922', '20130923',#2013
					'20120920', '20120923', '20120924',#2012
					'20110922', '20110925', '20110926',#2011
					'20100926', '20100927',#2010
					'20090927', '20090928']#2009
                    ,
                4:  ['20180927', '20180930', '20181001',#2018
                    '20170928', '20171001', '20171002',#2017
					'20160929', '20161002', '20161003',#2016
					'20151001', '20151004', '20151005',#2015
					'20140925', '20140928', '20140929',#2014
					'20130926', '20130929', '20130930',#2013
					'20120927', '20120930', '20121001',#2012
					'20110929', '20111002', '20111003',#2011
					'20101003', '20101004',#2010
					'20091004', '20091005']#2009
                    ,
                5:  ['20181004', '20181007', '20181008',#2018
                    '20171005', '20171008', '20171009',#2017
					'20161006', '20161009', '20161010',#2016
					'20151008', '20151011', '20151012',#2015
					'20141002', '20141005', '20141006',#2014
					'20131003', '20131006', '20131007',#2013
					'20121004', '20121007', '20121008',#2012
					'20111006', '20111009', '20111010',#2011
					'20101007', '20101010', '20101011',#2010
					'20091011', '20091012']#2009
                    ,
                6:  ['20181011', '20181014', '20181015',#2018
					'20171012', '20171015', '20171016',#2017
					'20161013', '20161016', '20161017',#2016
					'20151015', '20151018', '20151019',#2015
					'20141009', '20141012', '20141013',#2014
					'20131010', '20131013', '20131014',#2013
					'20121011', '20121014', '20121015',#2012
					'20111013', '20111016', '20111017',#2011
					'20101014', '20101017', '20101018',#2010
					'20091018', '20091019']#2009
                    ,
                7:  ['20181018', '20181021', '20181022',#2018
					'20171019', '20171022', '20171023',#2017
					'20161020', '20161023', '20161024',#2016
					'20151022', '20151025', '20151026',#2015
					'20141016', '20141019', '20141020',#2014
					'20131017', '20131020', '20131021',#2013
					'20121018', '20121021', '20121022',#2012
					'20111020', '20111023', '20111024',#2011
					'20101021', '20101024', '20101025',#2010
					'20091025', '20091026']#2009
                    ,
                8:  ['20181025', '20181028', '20181029',#2018
					'20171026', '20171029', '20171030',#2017
					'20161027', '20161030', '20161031',#2016
					'20151029', '20151101', '20151102',#2015
					'20141023', '20141026', '20141027',#2014
					'20131024', '20131027', '20131028',#2013
					'20121025', '20121028', '20121029',#2012
					'20111027', '20111030', '20111031',#2011
					'20101028', '20101031', '20101101',#2010
					'20091101', '20091102']#2009
                    ,
                9:  ['20181101', '20181104', '20181105',#2018
					'20171102', '20171105', '20171106',#2017
					'20161103', '20161106', '20161107',#2016
					'20151105', '20151108', '20151109',#2015
					'20141030', '20141102', '20141103',#2014
					'20131031', '20131103', '20131104',#2013
					'20121101', '20121104', '20121105',#2012
					'20111103', '20111106', '20111107',#2011
					'20101104', '20101107', '20101108',#2010
					'20091108', '20091109']#2009
                    ,
                10:  ['20181108', '20181111', '20181112',#2018
					 '20171109', '20171112', '20171113',#2017
					 '20161110', '20161113', '20161114',#2016
					 '20151112', '20151115', '20151116',#2015
					 '20141106', '20141109', '20141110',#2014
					 '20131107', '20131110', '20131111',#2013
					 '20121108', '20121111', '20121112',#2012
					 '20111110', '20111113', '20111114',#2011
					 '20101111', '20101114', '20101115',#2010
					 '20091112', '20091115', '20091116']#2009
                    ,
                11:  ['20181115', '20181118', '20181119',#2018
					 '20171116', '20171119', '20171120',#2017
					 '20161117', '20161120', '20161121',#2016
					 '20151119', '20151122', '20151123',#2015
					 '20141113', '20141116', '20141117',#2014
					 '20131114', '20131117', '20131118',#2013
					 '20121115', '20121118', '20121119',#2012
					 '20111117', '20111120', '20111121',#2011
					 '20101118', '20101121', '20101122',#2010
					 '20091119', '20091122', '20091123']#2009
                    ,
                12:  ['20181122', '20181125', '20181126',#2018
					 '20171123', '20171126', '20171127',#2017
					 '20161124', '20161127', '20161128',#2016
					 '20151126', '20151129', '20151130',#2015
					 '20141120', '20141123', '20141124',#2014
					 '20131121', '20131124', '20131125',#2013
					 '20121122', '20121125', '20121126',#2012
					 '20111124', '20111127', '20111128',#2011
					 '20101125', '20101128', '20101129',#2010
					 '20091126', '20091129', '20091130']#2009
                    ,
                13:  ['20181129', '20181202', '20181203',#2018
					 '20171130', '20171203', '20171204',#2017
					 '20161201', '20161204', '20161205',#2016
					 '20151203', '20151206', '20151207',#2015
					 '20141127', '20141130', '20141201',#2014
					 '20131128', '20131201', '20131202',#2013
					 '20121129', '20121202', '20121203',#2012
					 '20111201', '20111204', '20111205',#2011
					 '20101202', '20101205', '20101206',#2010
					 '20091203', '20091206', '20091207']#2009
                    ,
                14:  ['20181206', '20181209', '20181210',#2018
					 '20171207', '20171210', '20171211',#2017
					 '20161208', '20161211', '20161212',#2016
					 '20151210', '20151213', '20151214',#2015
					 '20141204', '20141207', '20141208',#2014
					 '20131205', '20131208', '20131209',#2013
					 '20121206', '20121209', '20121210',#2012
					 '20111208', '20111211', '20111212',#2011
					 '20101209', '20101212', '20101213',#2010
					 '20091210', '20091213', '20091214']#2009
                    ,
                15:  ['20181213', '20181215', '20181216', '20181217',#2018
					 '20171214', '20171216', '20171217', '20171218',#2017
					 '20161215', '20161217', '20161218', '20161219',#2016
					 '20151217', '20151219', '20151220', '20151221',#2015
					 '20141211', '20141213', '20141214', '20141215',#2014
					 '20131212', '20131214', '20131215', '20131216',#2013
					 '20121213', '20121216', '20121217',#2012
					 '20111215', '20111217', '20111218', '20111219',#2011
					 '20101216', '20101219', '20101220',#2010
					 '20091217', '20091219', '20091220', '20091221']#2009
                    ,
                16:  ['20181222', '20181223', '20181224',#2018
					 '20171223', '20171224', '20171225',#2017
					 '20161222', '20161224', '20161225', '20161226',#2016
					 '20151224', '20151226', '20151227', '20151228',#2015
					 '20141218', '20141220', '20141221', '20141222',#2014
					 '20131222', '20131223',#2013
					 '20121222', '20121223',#2012
					 '20111222', '20111224', '20111225', '20111226',#2011
					 '20101223', '20101225', '20101226', '20101227', '20101228',#2010
					 '20091225', '20091227', '20091228']#2009
                    ,
                17:  ['20181230',#2018
					 '20171231',#2017
					 '20170101',#2016
					 '20160103',#2015
					 '20141228',#2014
					 '20131229',#2013
					 '20121230',#2012
					 '20120101',#2011
					 '20110102',#2010
					 '20100103']#2009
					 
                }
    for key, value in week.items():#Check Dictionary Keys
        for i in range(len(value)):
            if str(x) == value[i]:
                return key
    return 0

num = 'test'
for idx in range(10):
    print(idx)
    if(idx < 9):
        num = str(idx + 10)
    else:
        num = '0'+str(idx)
    data = pd.read_csv("reg_pbp_20"+num+".csv", low_memory=False)

    row2 = 0
    found = False  
    
    kickoff = data.copy()
    kickoff = kickoff.loc[(kickoff['play_type'] == 'kickoff') & (kickoff['touchdown'] == 1)]
    kickoff = kickoff.groupby(['posteam', 'game_id', 'defteam'].apply(ko)
    
    kickoff.to_csv('ko_20'+num+'.csv')
    kickoff = pd.read_csv('ko_20'+num+'.csv', low_memory=False)
    
    punt = data.copy()
    punt = punt.loc[punt['play_type'] == 'punt']

    throw = data.copy()
    throw = throw.loc[(throw['play_type'] == 'pass') | (throw['play_type'] == 'qb_spike')]
    throw.loc[throw.sack == 1, 'yards_gained'] = 0
    throw['name'] = throw['passer_player_name']
    throw = throw.groupby(['passer_player_name', 'game_id', 'posteam', 'defteam', 'name']).apply(throws)

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