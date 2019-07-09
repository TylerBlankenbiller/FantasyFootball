from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import os

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

game = pd.read_csv("testLast4.csv", low_memory = False)
game['HCoach'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['ACoach'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['HDefense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['ADefense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['HOffense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['AOffense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')

game = game.loc[(game.SType == 'Regular') & (game.Week == 1) & (game.game_id == 2018090600)]
PO = game.loc[(game.posteam == 'PHI') ]
AO = game.loc[(game.posteam == 'ATL') ]

forcedFumble = PO['forced_fumble_player_1_player_name'].unique()
fumbleRecovery = PO['fumble_recovery_1_player_name'].unique()
fumble = PO['fumbled_1_player_name'].unique()
kicker = PO['kicker_player_name'].unique()
kReturner = PO['kickoff_returner_player_name'].unique()
pDefense = PO['pass_defense_1_player_name'].unique()
passer = PO['passer_player_name'].unique()
pReturner = PO['punt_returner_player_name'].unique()
punter = PO['punter_player_name'].unique()
qbHit = PO['qb_hit_1_player_name'].unique()
receiver = PO['receiver_player_name'].unique()
rusher = PO['rusher_player_name'].unique()
soloTackle = PO['solo_tackle_1_player_name'].unique()
tackle4Loss = PO['tackle_for_loss_1_player_name'].unique()
assistTackle = PO['assist_tackle_1_player_name'].unique()
interception = PO['interception_player_name'].unique()
weather = PO['Weather'].unique()
HCoach = PO['HCoach'].unique()
HDefense = PO['HDefense'].unique()
HOffense = PO['HOffense'].unique()

players = np.concatenate((forcedFumble, fumbleRecovery, fumble, kicker, kReturner, pDefense, passer,
            pReturner, punter, qbHit, fumbleRecovery, receiver, soloTackle, tackle4Loss, assistTackle,
            interception, weather, HCoach, HDefense, HOffense))
Pplayer = np.unique(str(players).split())
i = 0
delete = []
for x in np.nditer(Pplayer):
    Pplayer[i] = str(x).replace("'", '')
    print(Pplayer[i])
    if('0' in str(x)) or (']' in str(x)):
        delete.append(i)
    i+=1
Pplayer = np.delete(Pplayer, delete)




forcedFumble = AO['forced_fumble_player_1_player_name'].unique()
fumbleRecovery = AO['fumble_recovery_1_player_name'].unique()
fumble = AO['fumbled_1_player_name'].unique()
kicker = AO['kicker_player_name'].unique()
kReturner = AO['kickoff_returner_player_name'].unique()
pDefense = AO['pass_defense_1_player_name'].unique()
passer = AO['passer_player_name'].unique()
pReturner = AO['punt_returner_player_name'].unique()
punter = AO['punter_player_name'].unique()
qbHit = AO['qb_hit_1_player_name'].unique()
receiver = AO['receiver_player_name'].unique()
rusher = AO['rusher_player_name'].unique()
soloTackle = AO['solo_tackle_1_player_name'].unique()
tackle4Loss = AO['tackle_for_loss_1_player_name'].unique()
assistTackle = AO['assist_tackle_1_player_name'].unique()
interception = AO['interception_player_name'].unique()
weather = PO['Weather'].unique()
ACoach = PO['ACoach'].unique()
ADefense = PO['ADefense'].unique()
AOffense = PO['AOffense'].unique()

players = np.concatenate((forcedFumble, fumbleRecovery, fumble, kicker, kReturner, pDefense, passer,
            pReturner, punter, qbHit, fumbleRecovery, receiver, soloTackle, tackle4Loss, assistTackle,
            interception, weather, ACoach, ADefense, AOffense))
Aplayer = np.unique(str(players).split())

i = 0
delete = []
for x in np.nditer(Aplayer):
    Aplayer[i] = str(x).replace("'", '')
    print(Aplayer[i])
    if('0' in str(x)) or (']' in str(x)):
        delete.append(i)
    i+=1
Aplayer = np.delete(Aplayer, delete)
#print(Aplayer)

#print(type(interception))

game = pd.read_csv("testLast4.csv", low_memory = False)
game = game.loc[(game.SType == 'Pre') | (game.Year != 2018)]
print(len(game))
Pgame = game.loc[(game.forced_fumble_player_1_player_name.isin(Pplayer)) |
        (game['fumble_recovery_1_player_name'].isin(Pplayer)) |
        (game['fumbled_1_player_name'].isin(Pplayer)) |
        (game['kicker_player_name'].isin(Pplayer)) |
        (game['kickoff_returner_player_name'].isin(Pplayer)) |
        (game['pass_defense_1_player_name'].isin(Pplayer)) |
        (game['passer_player_name'].isin(Pplayer)) |
        (game['punt_returner_player_name'].isin(Pplayer)) |
        (game['punter_player_name'].isin(Pplayer)) |
        (game['qb_hit_1_player_name'].isin(Pplayer)) |
        (game['receiver_player_name'].isin(Pplayer)) |
        (game['rusher_player_name'].isin(Pplayer)) |
        (game['solo_tackle_1_player_name'].isin(Pplayer)) |
        (game['tackle_for_loss_1_player_name'].isin(Pplayer)) |
        (game['assist_tackle_1_player_name'].isin(Pplayer)) |
        (game['interception_player_name'].isin(Pplayer))]
        
        
print(len(Pgame))

Agame = game.loc[(game.forced_fumble_player_1_player_name.isin(Aplayer)) |
        (game['fumble_recovery_1_player_name'].isin(Aplayer)) |
        (game['fumbled_1_player_name'].isin(Aplayer)) |
        (game['kicker_player_name'].isin(Aplayer)) |
        (game['kickoff_returner_player_name'].isin(Aplayer)) |
        (game['pass_defense_1_player_name'].isin(Aplayer)) |
        (game['passer_player_name'].isin(Aplayer)) |
        (game['punt_returner_player_name'].isin(Aplayer)) |
        (game['punter_player_name'].isin(Aplayer)) |
        (game['qb_hit_1_player_name'].isin(Aplayer)) |
        (game['receiver_player_name'].isin(Aplayer)) |
        (game['rusher_player_name'].isin(Aplayer)) |
        (game['solo_tackle_1_player_name'].isin(Aplayer)) |
        (game['tackle_for_loss_1_player_name'].isin(Aplayer)) |
        (game['assist_tackle_1_player_name'].isin(Aplayer)) |
        (game['interception_player_name'].isin(Aplayer))]
print(len(Agame))
 
timeout = game.copy() 
#timeout = game.loc[(game.timeout == 1)]
#print(timeout)
#ntimeout = game.loc[(game.timeout == 0)]
#ntimeout = ntimeout.sample(n=len(timeout), random_state=1)
#timeout = pd.concat([timeout, ntimeout])
#print(timeout)
####################################################################################################
#    ___   _   _____   _
#   |   \ | | | ____| | |
#   | |\ \| | |  _|   | |__
#   |_|  \__| |_|     |____|    SIMULATOR
#
#####################################################################################################   
gameDF = pd.DataFrame({'SType':['Regular'], 'Weather':['Rain'],	'Week':1, 'Year':2019, 'air_yards':0,
        'assist_tackle_1_player_id':0, 'assist_tackle_1_player_name':'a', 'away_timeouts_remaining':3, 
        'complete_pass':0, 'defteam':'ATL', 'defteam_score':0, 'defteam_timeouts_remaining':3, 'down':1,
        'drive':1, 'extra_point_attempt':0, 'extra_point_result':0, 'field_goal_attempt':0,
        'field_goal_result':0, 'first_down_pass':0, 'first_down_rush':0, 'forced_fumble_player_1_player_id':0,
        'forced_fumble_player_1_player_name':'a', 'fourth_down_converted':0, 'fourth_down_failed':0,
        'fumble':0, 'fumble_lost':0, 'fumble_recovery_1_player_id':0, 'fumble_recovery_1_player_name':'a',
        'fumble_recovery_1_yards':0, 'fumbled_1_player_id':0, 'fumbled_1_player_name':'a',
        'game_id':0, 'game_seconds_remaining':3600, 'half_seconds_remaining':1800,
        'home_timeouts_remaining':3, 'incomplete_pass':0, 'interception':0, 'interception_player_id':0,
        'interception_player_name':'a',	'kick_distance':0, 'kicker_player_id':0, 'kicker_player_name':'a',
        'kickoff_returner_player_id':0, 'kickoff_returner_player_name':'a', 'pass_attempt':0,
        'pass_defense_1_player_id':0, 'pass_defense_1_player_name':'a', 'pass_location':0, 
        'pass_touchdown':0, 'passer_player_id':0, 'passer_player_name':'a', 'penalty_player_id':0,
        'penalty_player_name':'a', 'penalty_team':'a', 'penalty_yards':0, 'posteam':'PHI',
        'posteam_score':0, 'posteam_timeouts_remaining':3, 'posteam_type':'home', 'punt_attempt':0,
        'punt_blocked':0, 'punt_returner_player_id':0, 'punt_returner_player_name':'a',
        'punter_player_id':0, 'punter_player_name':'a', 'qb_hit':0, 'qb_hit_1_player_id':0,
        'qb_hit_1_player_name':'a',	'qb_kneel':0, 'qb_spike':0, 'qtr':1, 
        'quarter_seconds_remaining':900, 'receiver_player_id':0, 'receiver_player_name':'a',
        'return_yards':0, 'run_gap':'a', 'run_location':'a', 'rush_attempt':0, 'rush_touchdown':0,
        'rusher_player_id':0, 'rusher_player_name':'a', 'sack':0, 'safety':0, 'score_differential':0,
        'solo_tackle_1_player_id':0, 'solo_tackle_1_player_name':'a', 'tackle_for_loss_1_player_id':0,
        'tackle_for_loss_1_player_name':'a', 'third_down_converted':0, 'third_down_failed':0, 'timeout':0,
        'timeout_team':'a', 'total_away_score':0, 'total_home_score':0, 'touchback':0, 
        'two_point_attempt':0, 'two_point_conv_result':0, 'yardline_100':75, 'yards_after_catch':0,
        'yards_gained':0, 'ydstogo':10, 'HCoach':'Doug Pederson', 'HDefense':'Jim Schwartz',
        'HOffense':'Mike Groh', 'ACoach':'Dan Quinn', 'ADefense':'Marquand Manuel', 
        'AOffense':'Steve Sarkisian', 'totHit':0, 'totfirst_down_rush':0, 'totfirst_down_pass':0,
        'totincomplete_pass':0, 'totcomplete_pass':0, 'totinterception':0, 'totThirdDown_convert':0,
        'totFourthDown_convert':0, 'totThirdDown_fail':0, 'totFourthDown_fail':0, 'totPass_TD':0,
        'totRush_TD':0, 'totfumble':0, 'WSpeed':2, 'WDirection':'NW', 'WTemp':81, 'duration':0,
        'short':0, 'med':0, 'long':0, 'longest':0, 'attempt':0, 'acc':0})

gg = pd.read_csv('testLast4.csv', low_memory=False)
gg['HCoach'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['ACoach'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['HDefense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['ADefense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['HOffense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['AOffense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')

##########################################################################################################################################
#   Pass or Run
##########################################################################################################################################
passer = gg.copy()
passer = passer.loc[((passer.passer_player_name.isin(Pplayer) | (passer.rusher_player_name.isin(Pplayer))))]
def passering(training_df):
    '''
        Change stats that are strings into dummy columns
        These will be stats that are given and don't happen during the play.
        (I.E. Head Coach, Weather Type, Home Team, etc.)
    '''
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
    training_df['WDirection'] = training_df['WDirection'].astype(str)
    training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
    training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
    training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
    training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
    training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])

    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    training_df = training_df.drop(columns=['air_yards'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['complete_pass'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass', 'interception',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_defense_1_player_id', 'pass_defense_1_player_name', 'pass_location',
                'pass_touchdown', 'passer_player_id', 'passer_player_name', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id', 'receiver_player_name',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'yards_gained', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
gameDFPunteam = passering(gameDF)
        
passer = passering(passer)
col_list = (gameDFPunteam.append([gameDFPunteam,passer])).columns.tolist()
gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
passer = passer.loc[:, col_list].fillna(0)
X = passer.drop('pass_attempt', axis=1)
y = passer['pass_attempt']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=1000, random_state=1)

random_forest.fit(X_train, y_train)

y_predict = random_forest.predict(X_test)
print(y_predict)
print("Accuracy pass_attempt Attempt: ")
ab = accuracy_score(y_test, y_predict)

print(ab)

gameDFPunteam = gameDFPunteam.drop('pass_attempt', axis=1)
y_predict = random_forest.predict(gameDFPunteam)
print(y_predict)

print(len(passer.loc[(passer.pass_attempt == 1)]))
print(len(passer.loc[(passer.pass_attempt == 0)]))

gameDF['pass_attempt'] = y_predict

##########################################################################################################################################
#   Punt
##########################################################################################################################################
punt_attempt = gg.copy()
punt_attempt = punt_attempt.loc[(punt_attempt.punter_player_name.isin(Pplayer))]
nField = gg.copy()
print(nField.HCoach.unique())
print(Pplayer)
nField = nField.loc[(nField.HCoach.isin(Pplayer) & (nField.punt_attempt == 0))]
punt_attempt = pd.concat([punt_attempt, nField])
#field_goal_attempt = game.loc[(game.field_goal_attempt == 1)]
#print(field_goal_attempt)
#ntimeout = game.loc[(game.field_goal_attempt == 0)]
#ntimeout = ntimeout.sample(n=len(field_goal_attempt), random_state=1)
#field_goal_attempt = pd.concat([field_goal_attempt, ntimeout])
#print(field_goal_attempt)
def punt(training_df):
    '''
        Change stats that are strings into dummy columns
        These will be stats that are given and don't happen during the play.
        (I.E. Head Coach, Weather Type, Home Team, etc.)
    '''
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
    training_df['WDirection'] = training_df['WDirection'].astype(str)
    training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
    training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
    training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
    training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
    training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])

    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    training_df = training_df.drop(columns=['air_yards'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['complete_pass'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass', 'interception',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_attempt', 'pass_defense_1_player_id', 'pass_defense_1_player_name', 'pass_location',
                'pass_touchdown', 'passer_player_id', 'passer_player_name', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id', 'receiver_player_name',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'yards_gained', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
gameDFPunteam = punt(gameDF)
        
punt_attempt = punt(punt_attempt)
col_list = (gameDFPunteam.append([gameDFPunteam,punt_attempt])).columns.tolist()
gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
punt_attempt = punt_attempt.loc[:, col_list].fillna(0)
X = punt_attempt.drop('punt_attempt', axis=1)
y = punt_attempt['punt_attempt']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=1000, random_state=1)

random_forest.fit(X_train, y_train)

y_predict = random_forest.predict(X_test)
print(y_predict)
print("Accuracy Punt Attempt: ")
ab = accuracy_score(y_test, y_predict)

print(ab)

gameDFPunteam = gameDFPunteam.drop('punt_attempt', axis=1)
y_predict = random_forest.predict(gameDFPunteam)
print(y_predict)

print(len(punt_attempt.loc[(punt_attempt.punt_attempt == 1)]))
print(len(punt_attempt.loc[(punt_attempt.punt_attempt == 0)]))

gameDF['punt_attempt'] = y_predict

######################################################################################################
#
#    ___________   ____________
#   |____   ____| |   ______   |
#       |   |     |  |      |  |
#       |   |     |  |      |  |
#       |   |     |  |______|  |
#       |___|     |____________|    ~TIMEOUT
#
#######################################################################################################

def clean(training_df):
    '''
        Change stats that are strings into dummy columns
        These will be stats that are given and don't happen during the play.
        (I.E. Head Coach, Weather Type, Home Team, etc.)
    '''
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
    training_df['WDirection'] = training_df['WDirection'].astype(str)
    training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
    training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
    training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
    training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
    training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])

    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    #training_df = training_df.drop(columns=['air_yards'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    #training_df = training_df.drop(columns=['complete_pass'])
    #training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    #training_df = training_df.drop(columns=['field_goal_attempt'])
    training_df = training_df.drop(columns=['field_goal_result'])
    #training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
    #            'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
    #            'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
    #            'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass', 'interception',
    #            'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
    #            'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
    #            'pass_attempt', 'pass_defense_1_player_id', 'pass_defense_1_player_name', 'pass_location',
    #            'pass_touchdown', 'passer_player_id', 'passer_player_name', 'penalty_player_id', 'penalty_player_name',
    #            'penalty_team', 'penalty_yards', 'punt_attempt', 'punt_blocked', 'punt_returner_player_id',
    #            'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
    #            'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id', 'receiver_player_name',
    #            'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
    #            'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
    #            'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
    #            'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
    #            'yards_after_catch', 'yards_gained', 'duration'])
    training_df = training_df.drop(columns=['forced_fumble_player_1_player_id', 'forced_fumble_player_1_player_name', 'fumble_recovery_1_player_id',
                'fumble_recovery_1_player_name', 'game_id', 'fumbled_1_player_name', 'fumbled_1_player_id', 'interception_player_id',
                'interception_player_name', 'kicker_player_id', 'kickoff_returner_player_name', 'kickoff_returner_player_id', 'kicker_player_name',
                'pass_defense_1_player_id', 'pass_defense_1_player_name', 'pass_location', 'penalty_player_name', 'penalty_player_id', 'passer_player_name',
                'passer_player_id', 'penalty_team', 'punt_returner_player_id', 'qb_hit_1_player_id', 'punter_player_name', 'punter_player_id',
                'punt_returner_player_name', 'qb_hit_1_player_name', 'receiver_player_id', 'receiver_player_name', 'rusher_player_id', 'run_location',
                'run_gap', 'rusher_player_name', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name', 'tackle_for_loss_1_player_name', 
                'tackle_for_loss_1_player_id', 'timeout_team', 'two_point_conv_result'])
    return training_df

timeout = gg.copy() 
timeout = timeout.loc[(timeout.HCoach.isin(Pplayer))]
print(len(timeout))
gameDFTO = clean(gameDF)
timeout = clean(timeout)
col_list = (gameDFTO.append([gameDFTO,timeout])).columns.tolist()
gameDFTO = gameDFTO.loc[:, col_list].fillna(0)
timeout = timeout.loc[:, col_list].fillna(0)
X = timeout.drop('timeout', axis=1)
y = timeout['timeout']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=1000, random_state=1)

random_forest.fit(X_train, y_train)

y_predict = random_forest.predict(X_test)
print(y_predict)
print("Accuracy TO")
ab = accuracy_score(y_test, y_predict)

print(ab)

gameDFTO = gameDFTO.drop('timeout', axis=1)
y_predict = random_forest.predict(gameDFTO)
print(y_predict)

gameDF['timeout'] = y_predict


##########################################################################################################################################
#   Field Goal
##########################################################################################################################################
field_goal_attempt = gg.copy()
field = field_goal_attempt.loc[(field_goal_attempt.kicker_player_name.isin(Pplayer))]
nField = gg.copy()
nField = nField.loc[(nField.HCoach.isin(Pplayer) & (nField.field_goal_attempt == 0))]
field_goal_attempt = pd.concat([field_goal_attempt, nField])
#field_goal_attempt = game.loc[(game.field_goal_attempt == 1)]
#print(field_goal_attempt)
#ntimeout = game.loc[(game.field_goal_attempt == 0)]
#ntimeout = ntimeout.sample(n=len(field_goal_attempt), random_state=1)
#field_goal_attempt = pd.concat([field_goal_attempt, ntimeout])
#print(field_goal_attempt)
def fieldGoal(training_df):
    '''
        Change stats that are strings into dummy columns
        These will be stats that are given and don't happen during the play.
        (I.E. Head Coach, Weather Type, Home Team, etc.)
    '''
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
    training_df['WDirection'] = training_df['WDirection'].astype(str)
    training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
    training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
    training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
    training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
    training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])

    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    training_df = training_df.drop(columns=['air_yards'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['complete_pass'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass', 'interception',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_attempt', 'pass_defense_1_player_id', 'pass_defense_1_player_name', 'pass_location',
                'pass_touchdown', 'passer_player_id', 'passer_player_name', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_attempt', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id', 'receiver_player_name',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'yards_gained', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
gameDFTOTeam = fieldGoal(gameDF)
        
field_goal_attempt = fieldGoal(field_goal_attempt)
col_list = (gameDFTOTeam.append([gameDFTOTeam,field_goal_attempt])).columns.tolist()
gameDFTOTeam = gameDFTOTeam.loc[:, col_list].fillna(0)
field_goal_attempt = field_goal_attempt.loc[:, col_list].fillna(0)
X = field_goal_attempt.drop('field_goal_attempt', axis=1)
y = field_goal_attempt['field_goal_attempt']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=1000, random_state=1)

random_forest.fit(X_train, y_train)

y_predict = random_forest.predict(X_test)
print(y_predict)
print("Accuracy Field Goal Attempt: ")
ab = accuracy_score(y_test, y_predict)

print(ab)

gameDFTOTeam = gameDFTOTeam.drop('field_goal_attempt', axis=1)
y_predict = random_forest.predict(gameDFTOTeam)
print(y_predict)

print(len(field_goal_attempt.loc[(field_goal_attempt.field_goal_attempt == 1)]))
print(len(field_goal_attempt.loc[(field_goal_attempt.field_goal_attempt == 0)]))

gameDF['field_goal_attempt'] = y_predict
    
        
##########################################################################################################################################
#   field_goal_result
##########################################################################################################################################
field_goal_result = field.loc[(field.field_goal_attempt == 1)]
print(len(field_goal_result.loc[(field_goal_result.field_goal_result == 'made')]))
print(len(field_goal_result.loc[(field_goal_result.field_goal_result == 'missed')]))
#print(field_goal_result)
def kicker(training_df):
    '''
        Change stats that are strings into dummy columns
        These will be stats that are given and don't happen during the play.
        (I.E. Head Coach, Weather Type, Home Team, etc.)
    '''
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
    training_df['WDirection'] = training_df['WDirection'].astype(str)
    training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
    training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
    training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
    training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
    training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['kicker_player_id'])], axis=1)
    training_df = training_df.drop(columns=['kicker_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])

    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    training_df = training_df.drop(columns=['air_yards'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['complete_pass'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass', 'interception',
                'interception_player_id', 'interception_player_name', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_attempt', 'pass_defense_1_player_id', 'pass_defense_1_player_name', 'pass_location',
                'pass_touchdown', 'passer_player_id', 'passer_player_name', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_attempt', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id', 'receiver_player_name',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'yards_gained', 'duration', 'attempt', 'longest'])
    return training_df
    
gameDFTOTeam = kicker(gameDF)
        
field_goal_result = kicker(field_goal_result)
col_list = (gameDFTOTeam.append([gameDFTOTeam,field_goal_result])).columns.tolist()
gameDFTOTeam = gameDFTOTeam.loc[:, col_list].fillna(0)
field_goal_result = field_goal_result.loc[:, col_list].fillna(0)
X = field_goal_result.drop('field_goal_result', axis=1)
y = field_goal_result['field_goal_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

random_forest = RandomForestClassifier(n_estimators=20, max_depth=None, random_state=1)

random_forest.fit(X_train, y_train)

y_predict = random_forest.predict(X_test)
print(y_predict)
print("Accuracy Field Goal Result: ")
ab = accuracy_score(y_test, y_predict)
print(ab)

gameDFTOTeam = gameDFTOTeam.drop('field_goal_result', axis=1)
y_predict = random_forest.predict(gameDFTOTeam)
print(y_predict)

gameDF['field_goal_result'] = y_predict

