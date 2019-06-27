#include players
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('testLast2.csv', low_memory=False)

data = data.loc[(data['posteam'] != '0')]
data = data.loc[(data['Year'] == 2018) & (data['qtr'] < 3)]

teams = data['posteam'].unique()
np.savetxt('TeamNames.txt', teams, fmt='%s') 

#for team in teams:
if 1==1:
    fran = data
    #print(fran)
    data2 = fran[['Index', 'passer_player_id', 'receiver_player_id', 'rusher_player_id', 'kicker_player_id']].copy()
    
    data2['THRpasser_player_id'] = 'THR' + data2['passer_player_id']
    data2['RECreceiver_player_id'] = 'REC' + data2['receiver_player_id']
    data2['RSHrusher_player_id'] = 'RSH' + data2['rusher_player_id']
    data2['KICKkicker_player_id'] = 'KICK' + data2['kicker_player_id']
    
    QB = data2.loc[data2['passer_player_id'] != '0']
    QBs = QB['passer_player_id'].unique()
    np.savetxt('QB.txt', QBs, fmt='%s') 
    i = 0
    for QB in QBs:
        data2.loc[(data2.passer_player_id == QB) & (data2.passer_player_id != '0'), 'THRpasser_player_id'] = i+1
        data2.loc[data2.passer_player_id == '0', 'THRpasser_player_id'] = 0
        i+=1
        
    RB = data2.loc[data2['rusher_player_id'] != '0']
    RBs = RB['rusher_player_id'].unique()
    np.savetxt('RB.txt', RBs, fmt='%s')
    i = 0
    for RB in RBs:
        data2.loc[(data2.rusher_player_id == RB) & (data2.rusher_player_id != '0'), 'RSHrusher_player_id'] = i+1
        data2.loc[data2.rusher_player_id == '0', 'RSHrusher_player_id'] = 0
        i+=1
        
    WR = data2.loc[data2['receiver_player_id'] != '0']
    WRs = WR['receiver_player_id'].unique()
    np.savetxt('WR.txt', WRs, fmt='%s')
    i = 0
    for WR in WRs:
        data2.loc[(data2.receiver_player_id == WR) & (data2.receiver_player_id != '0'), 'RECreceiver_player_id'] = i+1
        data2.loc[data2.receiver_player_id == '0', 'RECreceiver_player_id'] = 0
        i+=1
        
    K = data2.loc[data2['kicker_player_id'] != '0']
    Ks = K['kicker_player_id'].unique()
    np.savetxt('K.txt', Ks, fmt='%s')
    i = 0
    for K in Ks:
        data2.loc[(data2.kicker_player_id == K) & (data2.kicker_player_id != '0'), 'KICKkicker_player_id'] = i+1
        data2.loc[data2.kicker_player_id == '0', 'KICKkicker_player_id'] = 0
        i+=1
        
    fran = pd.merge(fran, data2, left_on=['Index'], right_on=['Index'], how='left')
    
    fran.to_csv('players.csv', index=False)

