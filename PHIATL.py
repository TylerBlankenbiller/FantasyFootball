#Yards gained Run Away

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import statistics
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

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']
    
def build_model(train_dataset):
    model = keras.Sequential([
    layers.Dense(round(len(train_dataset.columns)*1.1), activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(round(len(train_dataset.columns)*0.5), activation=tf.nn.relu),
    layers.Dense(1)
        ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

home = 'PHI'
away = 'ATL'    

game = pd.read_csv("testLast4.csv", low_memory = False)
game['HCoach'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['ACoach'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['HDefense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['ADefense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['HOffense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
game['AOffense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')

game = game.loc[(game.SType == 'Regular') & (game.Week == 1) & (game.game_id == 2018090600)]
PO = game.loc[(game.posteam == home) ]
AO = game.loc[(game.posteam == away) ]

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
        'pass_defense_1_player_id':0, 'pass_defense_1_player_name':'0', 'pass_location':0, 
        'pass_touchdown':0, 'passer_player_id':0, 'passer_player_name':'a', 'penalty_player_id':0,
        'penalty_player_name':'a', 'penalty_team':'a', 'penalty_yards':0, 'posteam':'PHI',
        'posteam_score':0, 'posteam_timeouts_remaining':3, 'posteam_type':'home', 'punt_attempt':0,
        'punt_blocked':0, 'punt_returner_player_id':0, 'punt_returner_player_name':'a',
        'punter_player_id':0, 'punter_player_name':'a', 'qb_hit':0, 'qb_hit_1_player_id':0,
        'qb_hit_1_player_name':'a',	'qb_kneel':0, 'qb_spike':0, 'qtr':1, 
        'quarter_seconds_remaining':900, 'receiver_player_id':'0', 'receiver_player_name':'app',
        'return_yards':0, 'run_gap':'guard', 'run_location':'right', 'rush_attempt':0, 'rush_touchdown':0,
        'rusher_player_id':0, 'rusher_player_name':'sammy', 'sack':0, 'safety':0, 'score_differential':0,
        'solo_tackle_1_player_id':0, 'solo_tackle_1_player_name':'left', 'tackle_for_loss_1_player_id':0,
        'tackle_for_loss_1_player_name':'a', 'third_down_converted':0, 'third_down_failed':0, 'timeout':0,
        'timeout_team':'a', 'total_away_score':0, 'total_home_score':0, 'touchback':0, 
        'two_point_attempt':0, 'two_point_conv_result':0, 'yardline_100':75, 'yards_after_catch':0,
        'air_yards':0, 'ydstogo':10, 'HCoach':'Doug Pederson', 'HDefense':'Jim Schwartz',
        'HOffense':'Mike Groh', 'ACoach':'Dan Quinn', 'ADefense':'Marquand Manuel', 
        'AOffense':'Steve Sarkisian', 'totHit':0, 'totfirst_down_rush':0, 'totfirst_down_pass':0,
        'totincomplete_pass':0, 'totcomplete_pass':0, 'totinterception':0, 'totThirdDown_convert':0,
        'totFourthDown_convert':0, 'totThirdDown_fail':0, 'totFourthDown_fail':0, 'totPass_TD':0,
        'totRush_TD':0, 'totfumble':0, 'WSpeed':2, 'WDirection':'NW', 'WTemp':81, 'duration':0,
        'short':0, 'med':0, 'long':0, 'longest':0, 'attempt':0, 'acc':0, 'yards_gained':0})

gg = pd.read_csv('testLast4.csv', low_memory=False)
gg['HCoach'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['ACoach'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['HDefense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['ADefense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['HOffense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')
gg['AOffense'].replace(regex=True,inplace=True,to_replace=r' ',value=r'')

##########################################################################################################################################
#   Run Duration
##########################################################################################################################################
def Durations(training_df):
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
    training_df['run_location'] = 'RL' + training_df['run_location'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['rusher_player_name'])], axis=1)
    training_df = training_df.drop(columns=['rusher_player_name'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_gap'])], axis=1)
    training_df = training_df.drop(columns=['run_gap'])
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
                'return_yards', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predRunDuration(gameDF, rushPlayer):
    duration = gg.copy()
    print('run Duration XD')
    print(rushPlayer)
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        duration = duration.loc[((duration.rusher_player_name == rushPlayer)) | ((duration.ADefense.isin(Pplayer)) & (duration.Year == 2018) & (duration.rush_attempt == 1))]
    else:
        duration = duration.loc[((duration.rusher_player_name == rushPlayer)) | ((duration.HDefense.isin(Aplayer)) & (duration.Year == 2018) & (duration.rush_attempt == 1))]
    #temp = int.loc[(int.interception == 1)]
    #int = int.loc[(int.interception == 0)]
    #int= int.sample(n=len(temp), random_state=1)
    #print(len(int))
    #print(len(temp))
    #int = pd.concat([int, temp])
    print(len(duration))
    print(gameDF['posteam_type'])
    gameDFPunteam = Durations(gameDF)
            
    duration = Durations(duration)
    duration = duration.astype(float)
    gameDFPunteam = gameDFPunteam.astype(float)
    for col in duration.columns:
        if statistics.pstdev(duration[col])  <= 0.17:#0.07:#(aYards[col].mean() <= 0.01) | (aYards[col].mean() == 1):
            duration = duration.drop(columns=[col])
    common_cols = [col for col in set(duration.columns).intersection(gameDFPunteam.columns)]
    gameDFPunteam = gameDFPunteam[common_cols]
    col_list = (gameDFPunteam.append([gameDFPunteam,duration])).columns.tolist()
    duration = duration.loc[:, col_list].fillna(0)
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    duration.append(gameDFPunteam.iloc[0])
    gameDFPunteam.pop("duration")
            
    #print('test')
    #print(duration['duration'].mean())


    dataset = duration.copy()
    dataset.tail()

    train_dataset = dataset.sample(frac=0.75)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("duration")
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset.pop('duration')
    test_labels = test_dataset.pop('duration')


    normed_train_data = pd.DataFrame(columns=[])
    normed_test_data = pd.DataFrame(columns=[])
    
    for col in train_dataset.columns:
        normed_train_data[col] = (train_dataset[col] - train_dataset[col].mean()) / (train_dataset[col].max() - train_dataset[col].min())
        
    for col in test_dataset.columns:
        normed_test_data[col] = (test_dataset[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
        gameDFPunteam[col] = (gameDFPunteam[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
    
    #normed_train_data = (train_dataset - train_dataset.mean()) / (train_dataset.max() - train_dataset.min())#train_dataset#norm(train_dataset, train_stats)
    #normed_test_data = (test_dataset - test_dataset.mean()) / (test_dataset.max() - test_dataset.min())#test_dataset#norm(test_dataset, train_stats)

    cols = len(train_dataset.columns)

    EPOCHS = 1000
    model = build_model(train_dataset)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])


    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

    print("Testing set Mean Abs Error: {:5.2f} Duration".format(mae))


    test_predictions = model.predict(normed_test_data).flatten()

    #print(test_predictions)
    temp = model.predict(gameDFPunteam).flatten()
    print(temp[0])
    return(temp[0])


##########################################################################################################################################
#   Run aYards
##########################################################################################################################################
def rYardsGained(training_df):
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
    training_df['run_location'] = 'RL' + training_df['run_location'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['rusher_player_name'])], axis=1)
    training_df = training_df.drop(columns=['rusher_player_name'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_gap'])], axis=1)
    training_df = training_df.drop(columns=['run_gap'])
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
                'return_yards', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predRushYards(gameDF, rushPlayer):
    aYards = gg.copy()
    print("Rusher Name")
    print(rushPlayer)
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        aYards = aYards.loc[((aYards.rusher_player_name == rushPlayer)) | ((aYards.ADefense.isin(Aplayer)) & (aYards.Year == 2018) & (aYards.rush_attempt == 1))]
    else:
        aYards = aYards.loc[((aYards.rusher_player_name == rushPlayer)) | ((aYards.HDefense.isin(Pplayer)) & (aYards.Year == 2018) & (aYards.rush_attempt == 1))]# & (aYards.AOffense.isin(Aplayer)))]# | ((aYards.HDefense.isin(Pplayer)) & (aYards.Year == 2018) & (aYards.rush_attempt == 1))]
    print(len(aYards))
    print(gameDF['posteam_type'])
    print('My thing')
    #temp = int.loc[(int.interception == 1)]
    #int = int.loc[(int.interception == 0)]
    #int= int.sample(n=len(temp), random_state=1)
    #print(len(int))
    #print(len(temp))
    #int = pd.concat([int, temp])
    gameDFPunteam = rYardsGained(gameDF)
    print(len(aYards))
    print(gameDF['posteam_type'])
    aYards = rYardsGained(aYards)
    
    aYards = aYards.astype(float)
    gameDFPunteam = gameDFPunteam.astype(float)
    for col in aYards.columns:
        if statistics.pstdev(aYards[col])  <= 0.17:#0.07:#(aYards[col].mean() <= 0.01) | (aYards[col].mean() == 1):
            aYards = aYards.drop(columns=[col])
    common_cols = [col for col in set(aYards.columns).intersection(gameDFPunteam.columns)]
    gameDFPunteam = gameDFPunteam[common_cols]
    col_list = (gameDFPunteam.append([gameDFPunteam,aYards])).columns.tolist()
    aYards = aYards.loc[:, col_list].fillna(0)
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    aYards.append(gameDFPunteam.iloc[0])

    gameDFPunteam.pop("yards_gained")
    #print('test')
    #print(aYards['aYards'].mean())


    dataset = aYards.copy()
    dataset.tail()

    train_dataset = dataset.sample(frac=0.75)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("yards_gained")
    train_stats = train_stats.transpose()

    train_labels = train_dataset.pop('yards_gained')
    test_labels = test_dataset.pop('yards_gained')


    normed_train_data = pd.DataFrame(columns=[])
    normed_test_data = pd.DataFrame(columns=[])
    
    for col in train_dataset.columns:
        normed_train_data[col] = (train_dataset[col] - train_dataset[col].mean()) / (train_dataset[col].max() - train_dataset[col].min())
        
    for col in test_dataset.columns:
        normed_test_data[col] = (test_dataset[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
        gameDFPunteam[col] = (gameDFPunteam[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
    
    #normed_train_data = (train_dataset - train_dataset.mean()) / (train_dataset.max() - train_dataset.min())#train_dataset#norm(train_dataset, train_stats)
    #normed_test_data = (test_dataset - test_dataset.mean()) / (test_dataset.max() - test_dataset.min())#test_dataset#norm(test_dataset, train_stats)

    cols = len(train_dataset.columns)

    EPOCHS = 1000
    model = build_model(train_dataset)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])


    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

    print("Testing set Mean Abs Error: {:5.2f} B".format(mae))


    test_predictions = model.predict(normed_test_data).flatten()
    #print(test_predictions)
    temp = model.predict(gameDFPunteam).flatten()
    return(temp[0])

##########################################################################################################################################
#   run_gap
##########################################################################################################################################
def runGap(training_df):
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
    training_df['run_location'] = 'RL' + training_df['run_location'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['rusher_player_name'])], axis=1)
    training_df = training_df.drop(columns=['rusher_player_name'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
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
                'return_yards', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predRunGap(gameDF, rushPlayer):
    run_gap = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        run_gap = run_gap.loc[((run_gap.rusher_player_name == rushPlayer) & (run_gap.HOffense.isin(Pplayer))) | ((run_gap.ADefense.isin(Pplayer)) & (run_gap.Year == 2018) & (run_gap.rush_attempt == 1))]
    else:
        run_gap = run_gap.loc[((run_gap.rusher_player_name == rushPlayer) & (run_gap.AOffense.isin(Pplayer))) | ((run_gap.HDefense.isin(Pplayer)) & (run_gap.Year == 2018) & (run_gap.rush_attempt == 1))]
    #temp = int.loc[(int.interception == 1)]
    #int = int.loc[(int.interception == 0)]
    #int= int.sample(n=len(temp), random_state=1)
    #print(len(int))
    #print(len(temp))
    #int = pd.concat([int, temp])
    gameDFPunteam = runGap(gameDF)
            
    run_gap = runGap(run_gap)
    col_list = (gameDFPunteam.append([gameDFPunteam,run_gap])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    run_gap = run_gap.loc[:, col_list].fillna(0)
    X = run_gap.drop('run_gap', axis=1)
    y = run_gap['run_gap']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    #print(y_predict)
    #print("Accuracy run_gap Attempt: ")
    ab = accuracy_score(y_test, y_predict)

    #print(ab)
    gameDFPunteam = gameDFPunteam.drop('run_gap', axis=1)
    y_predict = random_forest.predict(gameDFPunteam)
    #print(y_predict)
    gameDF['run_gap'] = y_predict

##########################################################################################################################################
#   run_location
##########################################################################################################################################
def runLocation(training_df):
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['rusher_player_name'])], axis=1)
    training_df = training_df.drop(columns=['rusher_player_name'])
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
                'return_yards', 'run_gap', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predRunLocation(gameDF, rushPlayer):
    run_location = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        run_location = run_location.loc[((run_location.rusher_player_name == rushPlayer) & (run_location.HOffense.isin(Pplayer))) | ((run_location.ADefense.isin(Pplayer)) & (run_location.Year == 2018) & (run_location.rush_attempt == 1))]
    else:
        run_location = run_location.loc[((run_location.rusher_player_name == rushPlayer) & (run_location.AOffense.isin(Pplayer))) | ((run_location.HDefense.isin(Pplayer)) & (run_location.Year == 2018) & (run_location.rush_attempt == 1))]
    #temp = int.loc[(int.interception == 1)]
    #int = int.loc[(int.interception == 0)]
    #int= int.sample(n=len(temp), random_state=1)
    #print(len(int))
    #print(len(temp))
    #int = pd.concat([int, temp])
    gameDFPunteam = runLocation(gameDF)
            
    run_location = runLocation(run_location)
    col_list = (gameDFPunteam.append([gameDFPunteam,run_location])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    run_location = run_location.loc[:, col_list].fillna(0)
    X = run_location.drop('run_location', axis=1)
    y = run_location['run_location']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    #print(y_predict)
    #print("Accuracy run_location Attempt: ")
    ab = accuracy_score(y_test, y_predict)

    #print(ab)
    gameDFPunteam = gameDFPunteam.drop('run_location', axis=1)
    y_predict = random_forest.predict(gameDFPunteam)
    #print(y_predict)
    gameDF['run_location'] = y_predict

##########################################################################################################################################
#   rusher_player_name
##########################################################################################################################################
def rusherName(training_df):
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
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
                'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predRushPlayer(gameDF):
    rusher_player_name = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        rusher_player_name = rusher_player_name.loc[((rusher_player_name.rusher_player_name.isin(Pplayer)) & (rusher_player_name.Year == 2018))]
    else:
        rusher_player_name = rusher_player_name.loc[(rusher_player_name.rusher_player_name.isin(Aplayer)) & (rusher_player_name.Year == 2018)]
    #temp = int.loc[(int.interception == 1)]
    #int = int.loc[(int.interception == 0)]
    #int= int.sample(n=len(temp), random_state=1)
    #print(len(int))
    #print(len(temp))
    #int = pd.concat([int, temp])
    gameDFPunteam = rusherName(gameDF)
            
    rusher_player_name = rusherName(rusher_player_name)
    col_list = (gameDFPunteam.append([gameDFPunteam,rusher_player_name])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    rusher_player_name = rusher_player_name.loc[:, col_list].fillna(0)
    X = rusher_player_name.drop('rusher_player_name', axis=1)
    y = rusher_player_name['rusher_player_name']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    #print(y_predict)
    #print("Accuracy rusher_player_name Attempt: ")
    ab = accuracy_score(y_test, y_predict)

    #print(ab)
    gameDFPunteam = gameDFPunteam.drop('rusher_player_name', axis=1)
    y_predict = random_forest.predict(gameDFPunteam)
    #print(y_predict)
    gameDF['rusher_player_name'] = y_predict
    return(gameDF['rusher_player_name'])

##########################################################################################################################################
#   Pass Duration
##########################################################################################################################################
def PDurations(training_df):
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
    training_df['receiver_player_name'] = 'R' + training_df['receiver_player_name'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.pass_location!='middle', 'pass_location']='0'
    training_df.loc[training_df.pass_location=='middle', 'pass_location']='1'
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
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['field_goal_result'])
    #training_df = pd.concat([training_df, pd.get_dummies('Pass' + training_df['passer_player_name'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_name'])
    #training_df = pd.concat([training_df, pd.get_dummies('Rec' + training_df['receiver_player_name'])], axis=1)
    training_df = training_df.drop(columns=['receiver_player_name'])
    #training_df = pd.concat([training_df, pd.get_dummies('Defense' + training_df['pass_defense_1_player_name'])], axis=1)
    training_df = training_df.drop(columns=['pass_defense_1_player_name'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_defense_1_player_id',
                'pass_touchdown', 'passer_player_id', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predPassDuration(gameDF, receiverPlayer):
    passDuration = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        passDuration = passDuration.loc[((passDuration.receiver_player_name == receiverPlayer) & (passDuration.HOffense.isin(Pplayer))) | ((passDuration.ADefense.isin(Pplayer)) & (passDuration.Year == 2018) & (passDuration.pass_attempt == 1))]
    else:
        passDuration = passDuration.loc[((passDuration.receiver_player_name == receiverPlayer) & (passDuration.AOffense.isin(Pplayer))) | ((passDuration.HDefense.isin(Pplayer)) & (passDuration.Year == 2018) & (passDuration.pass_attempt == 1))]
        
    gameDFPunteam = PDurations(gameDF)
            
    passDuration = PDurations(passDuration)
    passDuration = passDuration.astype(float)
    gameDFPunteam = gameDFPunteam.astype(float)
    for col in passDuration.columns:
        if statistics.pstdev(passDuration[col])  <= 0.17:#0.07:#(aYards[col].mean() <= 0.01) | (aYards[col].mean() == 1):
            passDuration = passDuration.drop(columns=[col])
    common_cols = [col for col in set(passDuration.columns).intersection(gameDFPunteam.columns)]
    gameDFPunteam = gameDFPunteam[common_cols]
    col_list = (gameDFPunteam.append([gameDFPunteam,passDuration])).columns.tolist()
    passDuration = passDuration.loc[:, col_list].fillna(0)
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    passDuration.append(gameDFPunteam.iloc[0])
    gameDFPunteam.pop('duration')
            
    #print('test')
    #print(passDuration['duration'].mean())


    dataset = passDuration.copy()
    dataset.tail()

    train_dataset = dataset.sample(frac=0.75)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("duration")
    train_stats = train_stats.transpose()
    #print(train_stats)

    train_labels = train_dataset.pop('duration')
    test_labels = test_dataset.pop('duration')

    normed_train_data = pd.DataFrame(columns=[])
    normed_test_data = pd.DataFrame(columns=[])

    for col in train_dataset.columns:
        normed_train_data[col] = (train_dataset[col] - train_dataset[col].mean()) / (train_dataset[col].max() - train_dataset[col].min())
        
    for col in test_dataset.columns:
        normed_test_data[col] = (test_dataset[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
        gameDFPunteam[col] = (gameDFPunteam[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
    
    #normed_train_data = (train_dataset - train_dataset.mean()) / (train_dataset.max() - train_dataset.min())#train_dataset#norm(train_dataset, train_stats)
    #normed_test_data = (test_dataset - test_dataset.mean()) / (test_dataset.max() - test_dataset.min())#test_dataset#norm(test_dataset, train_stats)

    cols = len(train_dataset.columns)

    EPOCHS = 1000
    model = build_model(train_dataset)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])


    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

    #print("Testing set Mean Abs Error: {:5.2f} passDuration".format(mae))


    test_predictions = model.predict(normed_test_data).flatten()

    temp = model.predict(gameDFPunteam).flatten()
    print("Duration")
    print(temp[0])
    return(temp[0])


##########################################################################################################################################
#   yards_after_catch
##########################################################################################################################################
def afterCatch(training_df):
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
    training_df['receiver_player_name'] = 'R' + training_df['receiver_player_name'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.pass_location!='middle', 'pass_location']='0'
    training_df.loc[training_df.pass_location=='middle', 'pass_location']='1'
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
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
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['field_goal_result'])
    #training_df = pd.concat([training_df, pd.get_dummies('Pass' + training_df['passer_player_name'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_name'])
    #training_df = pd.concat([training_df, pd.get_dummies('Rec' + training_df['receiver_player_name'])], axis=1)
    training_df = training_df.drop(columns=['receiver_player_name'])
    #training_df = pd.concat([training_df, pd.get_dummies('Defense' + training_df['pass_defense_1_player_name'])], axis=1)
    training_df = training_df.drop(columns=['pass_defense_1_player_name'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_defense_1_player_id',
                'pass_touchdown', 'passer_player_id', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predYardsAfterCatch(gameDF, passPlayer, defPlayer, receiverPlayer):
    aYards = gg.copy()
    passPlayer = 'N.Foles'
    defPlayer = '0'
    receiverPlayer = 'N.Agholor'
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        aYards = aYards.loc[((aYards.passer_player_name == passPlayer) | (aYards.receiver_player_name == receiverPlayer) | (aYards.pass_defense_1_player_name == defPlayer)) & (aYards.pass_attempt == 1)]
    #temp = int.loc[(int.interception == 1)]
    #int = int.loc[(int.interception == 0)]
    #int= int.sample(n=len(temp), random_state=1)
    #print(len(int))
    #print(len(temp))
    #int = pd.concat([int, temp])
        
    gameDFPunteam = afterCatch(gameDF)
            
    aYards = afterCatch(aYards)
    
    aYards = aYards.astype(float)
    gameDFPunteam = gameDFPunteam.astype(float)
    for col in aYards.columns:
        if statistics.pstdev(aYards[col])  <= 0.17:#0.07:#(aYards[col].mean() <= 0.01) | (aYards[col].mean() == 1):
            aYards = aYards.drop(columns=[col])
    common_cols = [col for col in set(aYards.columns).intersection(gameDFPunteam.columns)]
    gameDFPunteam = gameDFPunteam[common_cols]
    col_list = (gameDFPunteam.append([gameDFPunteam,aYards])).columns.tolist()
    aYards = aYards.loc[:, col_list].fillna(0)
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    aYards.append(gameDFPunteam.iloc[0])

    gameDFPunteam.pop("yards_gained")
    #print('test')
    #print(aYards['aYards'].mean())


    dataset = aYards.copy()
    dataset.tail()

    train_dataset = dataset.sample(frac=0.75)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("yards_gained")
    train_stats = train_stats.transpose()

    train_labels = train_dataset.pop('yards_gained')
    test_labels = test_dataset.pop('yards_gained')


    normed_train_data = pd.DataFrame(columns=[])
    normed_test_data = pd.DataFrame(columns=[])
    
    for col in train_dataset.columns:
        normed_train_data[col] = (train_dataset[col] - train_dataset[col].mean()) / (train_dataset[col].max() - train_dataset[col].min())
        
    for col in test_dataset.columns:
        normed_test_data[col] = (test_dataset[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
        gameDFPunteam[col] = (gameDFPunteam[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
    
    #normed_train_data = (train_dataset - train_dataset.mean()) / (train_dataset.max() - train_dataset.min())#train_dataset#norm(train_dataset, train_stats)
    #normed_test_data = (test_dataset - test_dataset.mean()) / (test_dataset.max() - test_dataset.min())#test_dataset#norm(test_dataset, train_stats)

    cols = len(train_dataset.columns)

    EPOCHS = 1000
    model = build_model(train_dataset)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])


    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

    print("Testing set Mean Abs Error: {:5.2f} B".format(mae))


    test_predictions = model.predict(normed_test_data).flatten()
    #print(test_predictions)
    temp = model.predict(gameDFPunteam).flatten()
    return(temp[0])


##########################################################################################################################################
#   complete_pass
##########################################################################################################################################
def completion(training_df):
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
    training_df['receiver_player_name'] = 'R' + training_df['receiver_player_name'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.pass_location!='middle', 'pass_location']='0'
    training_df.loc[training_df.pass_location=='middle', 'pass_location']='1'
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
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
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = pd.concat([training_df, pd.get_dummies('Pass' + training_df['passer_player_name'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_name'])
    training_df = pd.concat([training_df, pd.get_dummies('Rec' + training_df['receiver_player_name'])], axis=1)
    training_df = training_df.drop(columns=['receiver_player_name'])
    #training_df = pd.concat([training_df, pd.get_dummies('Def21' + training_df['pass_defense_1_player_name'])], axis=1)
    training_df = training_df.drop(columns=['pass_defense_1_player_name'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_defense_1_player_id',
                'pass_touchdown', 'passer_player_id', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predCompletion(gameDF, passPlayer, defPlayer, receiverPlayer):
    complete = gg.copy()
    complete = complete.loc[((complete.pass_defense_1_player_name == defPlayer) | (complete.passer_player_name== passPlayer) | (complete.receiver_player_name== receiverPlayer)) & (complete.pass_attempt == 1) & (complete.interception == 0)]
    print('Completion')
    print(len(complete))
    #temp = int.loc[(int.interception == 1)]
    #int = int.loc[(int.interception == 0)]
    #int= int.sample(n=len(temp), random_state=1)
    #print(len(int))
    #print(len(temp))
    #int = pd.concat([int, temp])
        
    gameDFPunteam = completion(gameDF)
            
    complete = completion(complete)
    col_list = (gameDFPunteam.append([gameDFPunteam,complete])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    complete = complete.loc[:, col_list].fillna(0)
    X = complete.drop('complete_pass', axis=1)
    y = complete['complete_pass']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    #print(y_predict)
    #print("Accuracy complete_pass Attempt: ")
    ab = accuracy_score(y_test, y_predict)

    #print(ab)
    gameDFPunteam = gameDFPunteam.drop('complete_pass', axis=1)
    y_predict = random_forest.predict(gameDFPunteam)
    #print(y_predict)
    #print(len(complete.loc[((complete.complete_pass == 1))]))
    #print(len(complete.loc[((complete.complete_pass == 0))]))
    return(y_predict[0])

##########################################################################################################################################
#   interception
##########################################################################################################################################
def interception(training_df):
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
    training_df['receiver_player_name'] = 'R' + training_df['receiver_player_name'].astype(str)
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    
    training_df.loc[training_df.pass_location!='middle', 'pass_location']='0'
    training_df.loc[training_df.pass_location=='middle', 'pass_location']='1'
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
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
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['complete_pass'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = pd.concat([training_df, pd.get_dummies('Passer' + training_df['passer_player_name'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_name'])
    training_df = pd.concat([training_df, pd.get_dummies('Receiver' + training_df['receiver_player_name'])], axis=1)
    training_df = training_df.drop(columns=['receiver_player_name'])
    #training_df = pd.concat([training_df, pd.get_dummies('Defense' + training_df['pass_defense_1_player_name'])], axis=1)
    training_df = training_df.drop(columns=['pass_defense_1_player_name'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_defense_1_player_id',
                'pass_touchdown', 'passer_player_id', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predInterception(gameDF, passPlayer, defPlayer, receiverPlayer):
    int = gg.copy()
    int = int.loc[((int.pass_defense_1_player_name == defPlayer) | (int.passer_player_name== passPlayer) | (int.receiver_player_name == receiverPlayer)) & (int.pass_attempt == 1)]
    temp = int.loc[(int.interception == 1)]
    #int = int.loc[(int.interception == 0)]
    #int= int.sample(n=len(temp), random_state=1)
    #print(len(int))
    #print(len(temp))
    #int = pd.concat([int, temp])
    gameDFPunteam = interception(gameDF)
            
    int = interception(int)
    col_list = (gameDFPunteam.append([gameDFPunteam,int])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    int = int.loc[:, col_list].fillna(0)
    X = int.drop('interception', axis=1)
    y = int['interception']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    #print(y_predict)
    #print("Accuracy interception Attempt: ")
    ab = accuracy_score(y_test, y_predict)

    #print(ab)
    gameDFPunteam = gameDFPunteam.drop('interception', axis=1)
    y_predict = random_forest.predict(gameDFPunteam)
    #print(y_predict)
    #print(len(int.loc[((int.interception == 1))]))
    #print(len(int.loc[((int.interception == 0))]))
    gameDF['interception'] = y_predict
    return(gameDF['interception'])


##########################################################################################################################################
#   receiver_player_name
##########################################################################################################################################
def receiverName(training_df):
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
    
    training_df.loc[training_df.pass_location!='middle', 'pass_location']='0'
    training_df.loc[training_df.pass_location=='middle', 'pass_location']='1'
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
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
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['complete_pass'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['passer_player_name'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_name'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_defense_1_player_name'])], axis=1)
    training_df = training_df.drop(columns=['pass_defense_1_player_name'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass', 'interception',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_defense_1_player_id',
                'pass_touchdown', 'passer_player_id', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predReceiver(gameDF, passPlayer):
    receiver_player_name = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        receiver_player_name = receiver_player_name.loc[(receiver_player_name.receiver_player_name.isin(Pplayer)) & (receiver_player_name.passer_player_name == passPlayer) & (receiver_player_name.pass_attempt == 1)]
    else:    
        receiver_player_name = receiver_player_name.loc[(receiver_player_name.receiver_player_name.isin(Aplayer)) & (receiver_player_name.passer_player_name == passPlayer) & (receiver_player_name.pass_attempt == 1)]
    gameDFPunteam = receiverName(gameDF)
            
    receiver_player_name = receiverName(receiver_player_name)
    col_list = (gameDFPunteam.append([gameDFPunteam,receiver_player_name])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    receiver_player_name = receiver_player_name.loc[:, col_list].fillna(0)
    X = receiver_player_name.drop('receiver_player_name', axis=1)
    y = receiver_player_name['receiver_player_name']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    #print(y_predict)
    print("Accuracy receiver_player_name Attempt: ")
    ab = accuracy_score(y_test, y_predict)

    print(ab)
    gameDFPunteam = gameDFPunteam.drop('receiver_player_name', axis=1)
    y_predict = random_forest.predict(gameDFPunteam)
    print(y_predict)

    gameDF['receiver_player_name'] = y_predict
    return(gameDF['receiver_player_name'])


##########################################################################################################################################
#   Air Yards
##########################################################################################################################################
def airYards(training_df):
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
    
    training_df.loc[training_df.pass_location!='middle', 'pass_location']='0'
    training_df.loc[training_df.pass_location=='middle', 'pass_location']='1'
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
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
    training_df = training_df.drop(columns=['assist_tackle_1_player_id'])
    training_df = training_df.drop(columns=['assist_tackle_1_player_name'])
    training_df = training_df.drop(columns=['complete_pass'])
    training_df = training_df.drop(columns=['extra_point_attempt'])
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = pd.concat([training_df, pd.get_dummies('Pass' + training_df['passer_player_name'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_name'])
    #training_df = pd.concat([training_df, pd.get_dummies('Defense' + training_df['pass_defense_1_player_name'])], axis=1)
    training_df = training_df.drop(columns=['pass_defense_1_player_name'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass', 'interception',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_defense_1_player_id',
                'pass_touchdown', 'passer_player_id', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id', 'receiver_player_name',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predAirYards(gameDF, passPlayer, defPlayer):
    aYards = gg.copy()
    aYards = aYards.loc[((aYards.pass_defense_1_player_name == defPlayer) | (aYards.passer_player_name == passPlayer)) & (aYards.pass_attempt == 1)]
    gameDFPunteam = airYards(gameDF)
    aYards = airYards(aYards)
    aYards = aYards.astype(float)
    gameDFPunteam = gameDFPunteam.astype(float)
    for col in aYards.columns:
        if statistics.pstdev(aYards[col])  <= 0.17:#0.07:#(aYards[col].mean() <= 0.01) | (aYards[col].mean() == 1):
            aYards = aYards.drop(columns=[col])
    common_cols = [col for col in set(aYards.columns).intersection(gameDFPunteam.columns)]
    gameDFPunteam = gameDFPunteam[common_cols]
    col_list = (gameDFPunteam.append([gameDFPunteam,aYards])).columns.tolist()
    aYards = aYards.loc[:, col_list].fillna(0)
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    aYards.append(gameDFPunteam.iloc[0])

    gameDFPunteam.pop("air_yards")
    #print('test')
    #print(aYards['aYards'].mean())


    dataset = aYards.copy()
    dataset.tail()

    train_dataset = dataset.sample(frac=0.75)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("air_yards")
    train_stats = train_stats.transpose()

    train_labels = train_dataset.pop('air_yards')
    test_labels = test_dataset.pop('air_yards')


    normed_train_data = pd.DataFrame(columns=[])
    normed_test_data = pd.DataFrame(columns=[])
    
    for col in train_dataset.columns:
        normed_train_data[col] = (train_dataset[col] - train_dataset[col].mean()) / (train_dataset[col].max() - train_dataset[col].min())
        
    for col in test_dataset.columns:
        normed_test_data[col] = (test_dataset[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
        gameDFPunteam[col] = (gameDFPunteam[col] - test_dataset[col].mean()) / (test_dataset[col].max() - test_dataset[col].min())
    
    #normed_train_data = (train_dataset - train_dataset.mean()) / (train_dataset.max() - train_dataset.min())#train_dataset#norm(train_dataset, train_stats)
    #normed_test_data = (test_dataset - test_dataset.mean()) / (test_dataset.max() - test_dataset.min())#test_dataset#norm(test_dataset, train_stats)

    cols = len(train_dataset.columns)

    EPOCHS = 1000
    model = build_model(train_dataset)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])


    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

    print("Testing set Mean Abs Error: {:5.2f} B".format(mae))


    test_predictions = model.predict(normed_test_data).flatten()
    #print(test_predictions)
    temp = model.predict(gameDFPunteam).flatten()
    return(temp[0])


##########################################################################################################################################
#   pass_location
##########################################################################################################################################
def passLocation(training_df):
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
    
    training_df.loc[training_df.pass_location!='middle', 'pass_location']='0'
    training_df.loc[training_df.pass_location=='middle', 'pass_location']='1'
    training_df.loc[training_df.SType=='Regular', 'SType']='1'
    training_df.loc[training_df.SType=='Pre', 'SType']='0'
    training_df.loc[training_df.posteam_type=='home', 'posteam_type']='1'
    training_df.loc[training_df.posteam_type=='away', 'posteam_type']='0'
    training_df.loc[training_df.WTemp=='39/53', 'WTemp']='46'
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
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
    training_df = pd.concat([training_df, pd.get_dummies('Pass' + training_df['passer_player_name'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_name'])
    #training_df = pd.concat([training_df, pd.get_dummies('Def' + training_df['pass_defense_1_player_name'])], axis=1)
    training_df = training_df.drop(columns=['pass_defense_1_player_name'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass', 'interception',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_defense_1_player_id',
                'pass_touchdown', 'passer_player_id', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id', 'receiver_player_name',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predPassLocation(gameDF, passPlayer, defPlayer):
    pLocation = gg.copy()
    if(defPlayer != '0'):
        pLocation = pLocation.loc[((pLocation.pass_defense_1_player_name == defPlayer) | (pLocation.passer_player_name == passPlayer)) & (pLocation.pass_attempt == 1)]
    else:
        pLocation = pLocation.loc[(pLocation.passer_player_name == passPlayer) & (pLocation.pass_attempt == 1) & (pLocation.pass_defense_1_player_name == defPlayer)]
    #temp = int.loc[(int.interception == 1)]
    #int = int.loc[(int.interception == 0)]
    #int= int.sample(n=len(temp), random_state=1)
    #print(len(int))
    #print(len(temp))
    #int = pd.concat([int, temp])
    gameDFPunteam = passLocation(gameDF)
            
    pLocation = passLocation(pLocation)

    col_list = (gameDFPunteam.append([gameDFPunteam,pLocation])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    pLocation = pLocation.loc[:, col_list].fillna(0)
    X = pLocation.drop('pass_location', axis=1)
    y = pLocation['pass_location']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    #print(y_predict)
    #print("Accuracy pass_location Attempt: ")
    ab = accuracy_score(y_test, y_predict)

    #print(ab)
    gameDFPunteam = gameDFPunteam.drop('pass_location', axis=1)
    y_predict = random_forest.predict(gameDFPunteam)
    print('Pass location')
    print(y_predict)
    return(int(y_predict[0]))

##########################################################################################################################################
#   pass_defense_1_player_name
##########################################################################################################################################
def passDefense(training_df):
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['passer_player_name'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_name'])
    training_df = training_df.drop(columns=['first_down_pass', 'first_down_rush', 'forced_fumble_player_1_player_id',
                'forced_fumble_player_1_player_name', 'fourth_down_converted', 'fourth_down_failed', 'fumble',
                'fumble_lost', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name', 'fumble_recovery_1_yards',
                'fumbled_1_player_id', 'fumbled_1_player_name', 'game_id', 'incomplete_pass', 'interception',
                'interception_player_id', 'interception_player_name', 'kick_distance', 'kicker_player_id', 
                'kicker_player_name', 'kickoff_returner_player_id', 'kickoff_returner_player_name',
                'pass_defense_1_player_id', 'pass_location',
                'pass_touchdown', 'passer_player_id', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id', 'receiver_player_name',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predDefPlayer(gameDF, passPlayer):
    pDefense = gg.copy()
    pDefense = pDefense.loc[((pDefense.pass_defense_1_player_name.isin(Pplayer)) | ((pDefense.passer_player_name == passPlayer) & (pDefense.pass_defense_1_player_name == '0'))) & (pDefense.pass_attempt == 1)]
    gameDFPunteam = passDefense(gameDF)
            
    pDefense = passDefense(pDefense)
    col_list = (gameDFPunteam.append([gameDFPunteam,pDefense])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    pDefense = pDefense.loc[:, col_list].fillna(0)
    X = pDefense.drop('pass_defense_1_player_name', axis=1)
    y = pDefense['pass_defense_1_player_name']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    #print(y_predict)
    print("Accuracy pass_defense_1_player_name Attempt: ")
    ab = accuracy_score(y_test, y_predict)

    print(ab)

    gameDFPunteam = gameDFPunteam.drop('pass_defense_1_player_name', axis=1)
    y_predict = random_forest.predict(gameDFPunteam)
    print(y_predict)

    gameDF['pass_defense_1_player_name'] = y_predict
    return(int(y_predict[0]))

##########################################################################################################################################
#   Pass Player
##########################################################################################################################################
def passeringp(training_df):
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
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
                'pass_touchdown', 'passer_player_id', 'penalty_player_id', 'penalty_player_name',
                'penalty_team', 'penalty_yards', 'punt_blocked', 'punt_returner_player_id',
                'punt_returner_player_name', 'punter_player_id', 'punter_player_name', 'qb_hit', 'qb_hit_1_player_id',
                'qb_hit_1_player_name', 'qb_kneel', 'qb_spike', 'receiver_player_id', 'receiver_player_name',
                'return_yards', 'run_gap', 'run_location', 'rush_attempt', 'rush_touchdown', 'rusher_player_id',
                'rusher_player_name', 'sack', 'safety', 'solo_tackle_1_player_id', 'solo_tackle_1_player_name',
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 'third_down_converted',
                'third_down_failed', 'timeout_team', 'touchback', 'two_point_attempt', 'two_point_conv_result',
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predPassPlayer(gameDF):
    passer = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        passer = passer.loc[((passer.passer_player_name.isin(Pplayer)) & (passer.Year == 2018))]
    else:
        passer = passer.loc[((passer.passer_player_name.isin(Aplayer)) & (passer.Year == 2018))]
    gameDFPunteam = passeringp(gameDF)
            
    passer = passeringp(passer)
    col_list = (gameDFPunteam.append([gameDFPunteam,passer])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    passer = passer.loc[:, col_list].fillna(0)
    y = passer['passer_player_name']
    X = passer.drop('passer_player_name', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    #print(y_predict)
    print("Accuracy passer_player_name Attempt: ")
    ab = accuracy_score(y_test, y_predict)

    print(ab)

    gameDFPunteam = gameDFPunteam.drop('passer_player_name', axis=1)
    y_predict = random_forest.predict(gameDFPunteam)
    print(y_predict)

    return(y_predict[0])

##########################################################################################################################################
#   Pass or Run
##########################################################################################################################################
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
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
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predPassRun(gameDF):
    passer = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        passer = passer.loc[((passer.passer_player_name.isin(Pplayer) | (passer.rusher_player_name.isin(Pplayer))) & (passer.HCoach.isin(Pplayer))) & (passer.qtr == gameDF['qtr'].iloc[0])]
    else:
        print('Away PassRun')
        passer = passer.loc[((passer.passer_player_name.isin(Aplayer) | (passer.rusher_player_name.isin(Aplayer)))) & (passer.qtr == gameDF['qtr'].iloc[0])]
    gameDFPunteam = passering(gameDF)
            
    if(len(passer.loc[passer['pass_attempt'] == 1]) > len(passer.loc[passer['rush_attempt'] == 1])):
        temp = passer.loc[passer['rush_attempt'] == 1]
        temp2 = passer.loc[passer['pass_attempt'] == 1]
        temp2 = temp2.sample(n = len(passer.loc[passer['rush_attempt'] == 1])) 
        passer = pd.concat([temp, temp2])
    else:
        temp = passer.loc[passer['pass_attempt'] == 1]
        temp2 = passer.loc[passer['rush_attempt'] == 1]
        temp2 = temp2.sample(n = len(passer.loc[passer['pass_attempt'] == 1])) 
        passer = pd.concat([temp, temp2])
    
    passer = passering(passer)
    col_list = (gameDFPunteam.append([gameDFPunteam,passer])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    passer = passer.loc[:, col_list].fillna(0)
    X = passer.drop('pass_attempt', axis=1)
    y = passer['pass_attempt']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None)

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

    return(int(y_predict[0]))

##########################################################################################################################################
#   Punt
##########################################################################################################################################
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
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df
    
def predPunt(gameDF):
    punt_attempt = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        punt_attempt = punt_attempt.loc[(punt_attempt.punter_player_name.isin(Pplayer))]
    else:
        punt_attempt = punt_attempt.loc[(punt_attempt.punter_player_name.isin(Aplayer))]
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
    gameDFPunteam = punt(gameDF)
            
    punt_attempt = punt(punt_attempt)
    col_list = (gameDFPunteam.append([gameDFPunteam,punt_attempt])).columns.tolist()
    gameDFPunteam = gameDFPunteam.loc[:, col_list].fillna(0)
    punt_attempt = punt_attempt.loc[:, col_list].fillna(0)
    X = punt_attempt.drop('punt_attempt', axis=1)
    y = punt_attempt['punt_attempt']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=1000)

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

    return(int(y_predict[0]))

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
    #            'yards_after_catch', 'aYards', 'duration'])
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

X_train, X_test, y_train, y_test = train_test_split(X, y)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=1000)

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
                'yards_after_catch', 'duration', 'short', 'med', 'long', 'longest', 'attempt'])
    return training_df

def predFieldGoal(gameDF):
    field_goal_attempt = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        field = field_goal_attempt.loc[(field_goal_attempt.kicker_player_name.isin(Pplayer))]
    else:
        field = field_goal_attempt.loc[(field_goal_attempt.kicker_player_name.isin(Aplayer))]
    nField = gg.copy()
    nField = nField.loc[(nField.HCoach.isin(Pplayer) & (nField.field_goal_attempt == 0))]
    field_goal_attempt = pd.concat([field_goal_attempt, nField])
    #field_goal_attempt = game.loc[(game.field_goal_attempt == 1)]
    #print(field_goal_attempt)
    #ntimeout = game.loc[(game.field_goal_attempt == 0)]
    #ntimeout = ntimeout.sample(n=len(field_goal_attempt), random_state=1)
    #field_goal_attempt = pd.concat([field_goal_attempt, ntimeout])
    #print(field_goal_attempt)    
    gameDFTOTeam = fieldGoal(gameDF)
            
    field_goal_attempt = fieldGoal(field_goal_attempt)
    col_list = (gameDFTOTeam.append([gameDFTOTeam,field_goal_attempt])).columns.tolist()
    gameDFTOTeam = gameDFTOTeam.loc[:, col_list].fillna(0)
    field_goal_attempt = field_goal_attempt.loc[:, col_list].fillna(0)
    X = field_goal_attempt.drop('field_goal_attempt', axis=1)
    y = field_goal_attempt['field_goal_attempt']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=1000)

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

    return(int(y_predict[0]))
    
        
##########################################################################################################################################
#   field_goal_result
##########################################################################################################################################
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
                'yards_after_catch', 'duration', 'attempt', 'longest'])
    return training_df
    
def predFGResult(gameDF):
    field_goal_attempt = gg.copy()
    if(gameDF['posteam_type'].iloc[0] == 'home'):
        field = field_goal_attempt.loc[(field_goal_attempt.kicker_player_name.isin(Pplayer))]
    else:
        field = field_goal_attempt.loc[(field_goal_attempt.kicker_player_name.isin(Pplayer))]
    nField = gg.copy()
    nField = nField.loc[(nField.HCoach.isin(Pplayer) & (nField.field_goal_attempt == 0))]
    field_goal_attempt = pd.concat([field_goal_attempt, nField])
    field_goal_result = field.loc[(field.field_goal_attempt == 1)]
    print(len(field_goal_result.loc[(field_goal_result.field_goal_result == 'made')]))
    print(len(field_goal_result.loc[(field_goal_result.field_goal_result == 'missed')]))
    #print(field_goal_result)
    
    gameDF['short'] = field_goal_result['short'].iloc[0]
    gameDF['med'] = field_goal_result['med'].iloc[0]
    gameDF['long'] = field_goal_result['long'].iloc[0]
    gameDF['longest'] = field_goal_result['longest'].iloc[0]
    gameDF['attempt'] = field_goal_result['attempt'].iloc[0]
    gameDF['acc'] = field_goal_result['acc'].iloc[0]
    
    gameDFTOTeam = kicker(gameDF)
            
    field_goal_result = kicker(field_goal_result)
    col_list = (gameDFTOTeam.append([gameDFTOTeam,field_goal_result])).columns.tolist()
    gameDFTOTeam = gameDFTOTeam.loc[:, col_list].fillna(0)
    field_goal_result = field_goal_result.loc[:, col_list].fillna(0)
    X = field_goal_result.drop('field_goal_result', axis=1)
    y = field_goal_result['field_goal_result']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(n_estimators=20, max_depth=None)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print(y_predict)
    print("Accuracy Field Goal Result: ")
    ab = accuracy_score(y_test, y_predict)
    print(ab)

    gameDFTOTeam = gameDFTOTeam.drop('field_goal_result', axis=1)
    y_predict = random_forest.predict(gameDFTOTeam)
    print(y_predict)
    print(gameDF['kick_distance'])
    print(len(gameDF['kick_distance']))
    print('kick distance')
    print(field_goal_result['short'].iloc[0])
    gameDF['field_goal_result'] = y_predict
    print(gameDF)
    return(gameDF['field_goal_result'])

    
justScored = 0
hHit = 0
aHit = 0
hFirstRush = 0
aFirstRush = 0
hFirstPass = 0  #
aFirstPass = 0  #
hIncomplete = 0 #
aIncomplete = 0 #
hComplete = 0   #
aComplete = 0   #
hInt = 0        #
aInt = 0        #
h3rdConvert = 0 #?
a3rdConvert = 0 #?
h4thConvert = 0 #?
a4thConvert = 0 #?
h3rdFail = 0    #?
a3rdFail = 0    #?
h4thFail = 0    #?
a4thFail = 0    #?
hPassTD = 0     #
aPassTD = 0     #
hRushTD = 0
aRushTD = 0
hFumble = 0
aFumble = 0


passes = 0
rushes = 0
while(gameDF['qtr'].iloc[0] < 5):
    #Kick a field Goal?
    gameDF['field_goal_attempt'] = predFieldGoal(gameDF)
    gameDF['punt_attempt'] = predPunt(gameDF)
    if(passes >= 2):
        passes = 0
        gameDF['pass_attempt'] = 0
    elif(rushes >= 2):
        rushes = 0
        gameDF['pass_attempt'] = 1
    else:
        gameDF['pass_attempt'] = predPassRun(gameDF)
    print("PAss Attmpt")
    print(gameDF['pass_attempt'].iloc[0])
    #################################################################################################################################
    #
    #   FIELD GOAL
    #
    #################################################################################################################################
    if(int(gameDF['field_goal_attempt'].iloc[0]) == 1):
        print('FIELD GOAL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #Check if Field Goal is Good
        gameDF['kick_distance'] = gameDF['yardline_100'] + 18
        gameDF['field_goal_result'] = predFGResult(gameDF)
        if(gameDF['field_goal_result'].iloc[0] == 'made'):
            #Add points for Field Goal
            gameDF['posteam_score'] += 3
            gameDF['score_differential'] = gameDF['defteam_score'] - gameDF['posteam_score']
            if(gameDF['posteam_type'].iloc[0] == 'home'):
                gameDF['total_home_score'] += 3
            else:
                gameDF['total_away_score'] += 3
            gameDF['yardline_100'] = 75
        else:
            gameDF['yardline_100'] = 100 - gameDF['yardline_100']
            
        #Change Score, Position, Timeouts, Drive
        #Posteam Type
        if(gameDF['posteam_type'].iloc[0] == 'home'):
            gameDF['posteam_type'] = 'away'####################################################################################
            gameDF['totinterception'] = aInt
            gameDF['totincomplete_pass'] = aIncomplete
            gameDF['totcomplete_pass'] = aComplete
            gameDF['totThirdDown_convert'] = a3rdConvert
            gameDF['totThirdDown_fail'] = a3rdFail
            gameDF['totFourthDown_convert'] = a4thConvert
            gameDF['totFourthDown_fail'] = a4thFail
            gameDF['totPass_TD'] = aPassTD
            gameDF['totRush_TD'] = aRushTD
            gameDF['totfumble'] = aFumble
            gameDF['totHit'] = aHit
            gameDF['totfirst_down_rush'] = aFirstRush
            gameDF['totfirst_down_pass'] = aFirstPass
            gameDF['posteam_type'] = 'away'
            #Change teams (Punt)
            #Score
            temp = gameDF['posteam_score'].iloc[0]
            gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
            gameDF['defteam_score'] = temp
                
            #Possession
            temp = gameDF['posteam'].iloc[0]
            gameDF['posteam'] = gameDF['defteam'].iloc[0]
            gameDF['defteam'] = temp
            
            print("Change")
            print(gameDF['posteam'].iloc[0])
            print(gameDF['defteam'].iloc[0])
            print(temp)
                
            #Timeouts
            temp = gameDF['posteam_timeouts_remaining'].iloc[0]
            gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
            gameDF['defteam_timeouts_remaining'] = temp
            
            gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                
            #Drive
            gameDF['drive'] += 1#############################################################################################
        else:
            gameDF['posteam_type'] = 'home'###################################################################################
            gameDF['totinterception'] = hInt
            gameDF['totincomplete_pass'] = hIncomplete
            gameDF['totcomplete_pass'] = hComplete
            gameDF['totThirdDown_convert'] = h3rdConvert
            gameDF['totThirdDown_fail'] = h3rdFail
            gameDF['totFourthDown_convert'] = h4thConvert
            gameDF['totFourthDown_fail'] = h4thFail
            gameDF['totPass_TD'] = hPassTD
            gameDF['totRush_TD'] = hRushTD
            gameDF['totfumble'] = hFumble
            gameDF['totHit'] = hHit
            gameDF['totfirst_down_rush'] = hFirstRush
            gameDF['totfirst_down_pass'] = hFirstPass
            gameDF['posteam_type'] = 'home'
            #Score
            temp = gameDF['posteam_score'].iloc[0]
            gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
            gameDF['defteam_score'] = temp
                
            #Possession
            temp = gameDF['posteam'].iloc[0]
            gameDF['posteam'] = gameDF['defteam'].iloc[0]
            gameDF['defteam'] = temp
            
            print("Change")
            print(gameDF['posteam'].iloc[0])
            print(gameDF['defteam'].iloc[0])
            print(temp)
                
            #Timeouts
            temp = gameDF['posteam_timeouts_remaining'].iloc[0]
            gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
            gameDF['defteam_timeouts_remaining'] = temp
            
            gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                
            #Drive
            gameDF['drive'] += 1#####################################################################################################
        #Reset Stuff
        gameDF['ydstogo'] = 10
        gameDF['down'] = 1
       
        #Take off time
        gameDF['game_seconds_remaining'] -= 5
        gameDF['half_seconds_remaining'] -= 5
        gameDF['quarter_seconds_remaining'] -= 5
        if(gameDF['quarter_seconds_remaining'].iloc[0] <= 0):
            gameDF['qtr'] += 1
            gameDF['quarter_seconds_remaining'] = 900
        if(gameDF['half_seconds_remaining'].iloc[0] <= 0):
            gameDF['half_seconds_remaining'] = 1800
            gameDF['game_seconds_remaining'] = 1800
    #################################################################################################################################
    #
    #   PUNT
    #
    #################################################################################################################################
    elif(int(gameDF['punt_attempt'].iloc[0]) == 1):
        print('PUNT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        gameDF['yardline_100'] = 100-(gameDF['yardline_100']-44)
        if(gameDF['yardline_100'].iloc[0] <= 0):
            gameDF['yardline_100'] = 80
            
        #Change Score, Position, Timeouts, Drive
        #Posteam Type
        if(gameDF['posteam_type'].iloc[0] == 'home'):
            gameDF['posteam_type'] = 'away'####################################################################################
            gameDF['totinterception'] = aInt
            gameDF['totincomplete_pass'] = aIncomplete
            gameDF['totcomplete_pass'] = aComplete
            gameDF['totThirdDown_convert'] = a3rdConvert
            gameDF['totThirdDown_fail'] = a3rdFail
            gameDF['totFourthDown_convert'] = a4thConvert
            gameDF['totFourthDown_fail'] = a4thFail
            gameDF['totPass_TD'] = aPassTD
            gameDF['totRush_TD'] = aRushTD
            gameDF['totfumble'] = aFumble
            gameDF['totHit'] = aHit
            gameDF['totfirst_down_rush'] = aFirstRush
            gameDF['totfirst_down_pass'] = aFirstPass
            gameDF['posteam_type'] = 'away'
            #Change teams (Punt)
            #Score
            temp = gameDF['posteam_score'].iloc[0]
            gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
            gameDF['defteam_score'] = temp
                
            #Possession
            temp = gameDF['posteam'].iloc[0]
            gameDF['posteam'] = gameDF['defteam'].iloc[0]
            gameDF['defteam'] = temp
            
            print("Change")
            print(gameDF['posteam'].iloc[0])
            print(gameDF['defteam'].iloc[0])
            print(temp)
                
            #Timeouts
            temp = gameDF['posteam_timeouts_remaining'].iloc[0]
            gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
            gameDF['defteam_timeouts_remaining'] = temp
            
            gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                
            #Drive
            gameDF['drive'] += 1#############################################################################################
        else:
            gameDF['posteam_type'] = 'home'###################################################################################
            gameDF['totinterception'] = hInt
            gameDF['totincomplete_pass'] = hIncomplete
            gameDF['totcomplete_pass'] = hComplete
            gameDF['totThirdDown_convert'] = h3rdConvert
            gameDF['totThirdDown_fail'] = h3rdFail
            gameDF['totFourthDown_convert'] = h4thConvert
            gameDF['totFourthDown_fail'] = h4thFail
            gameDF['totPass_TD'] = hPassTD
            gameDF['totRush_TD'] = hRushTD
            gameDF['totfumble'] = hFumble
            gameDF['totHit'] = hHit
            gameDF['totfirst_down_rush'] = hFirstRush
            gameDF['totfirst_down_pass'] = hFirstPass
            gameDF['posteam_type'] = 'home'
            #Score
            temp = gameDF['posteam_score'].iloc[0]
            gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
            gameDF['defteam_score'] = temp
                
            #Possession
            temp = gameDF['posteam'].iloc[0]
            gameDF['posteam'] = gameDF['defteam'].iloc[0]
            gameDF['defteam'] = temp
            
            print("Change")
            print(gameDF['posteam'].iloc[0])
            print(gameDF['defteam'].iloc[0])
            print(temp)
                
            #Timeouts
            temp = gameDF['posteam_timeouts_remaining'].iloc[0]
            gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
            gameDF['defteam_timeouts_remaining'] = temp
            
            gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                
            #Drive
            gameDF['drive'] += 1#####################################################################################################
        #Reset Stuff
        gameDF['ydstogo'] = 10
        gameDF['down'] = 1
       
        #Take off time
        gameDF['game_seconds_remaining'] -= 12
        gameDF['half_seconds_remaining'] -= 12
        gameDF['quarter_seconds_remaining'] -= 12
        if(gameDF['quarter_seconds_remaining'].iloc[0] <= 0):
            gameDF['qtr'] += 1
            gameDF['quarter_seconds_remaining'] = 900
        if(gameDF['half_seconds_remaining'].iloc[0] <= 0):
            gameDF['half_seconds_remaining'] = 1800
            gameDF['game_seconds_remaining'] = 1800
    #################################################################################################################################
    #
    #   PASS
    #
    #################################################################################################################################
    elif(int(gameDF['pass_attempt'].iloc[0]) == 1):
        passes+=1
        rushes = 0
        print('PASS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #Pred Stats
        gameDF['passer_player_name'] = predPassPlayer(gameDF)
        gameDF['pass_defense_1_player_name'] = predDefPlayer(gameDF, gameDF['passer_player_name'].iloc[0])
        gameDF['pass_location'] = predPassLocation(gameDF, gameDF['passer_player_name'].iloc[0], gameDF['pass_defense_1_player_name'].iloc[0])
        gameDF['air_yards'] = predAirYards(gameDF, gameDF['passer_player_name'].iloc[0], gameDF['pass_defense_1_player_name'].iloc[0])
        gameDF['receiver_player_name'] = predReceiver(gameDF, gameDF['passer_player_name'].iloc[0])
        gameDF['interception'] = predInterception(gameDF, gameDF['passer_player_name'].iloc[0], gameDF['pass_defense_1_player_name'].iloc[0], gameDF['receiver_player_name'].iloc[0])
        #Intercepted!
        if(int(gameDF['interception'].iloc[0]) == 1):
            print('Picked OFF!!!')
            gameDF['yardline_100'] = 100 - gameDF['yardline_100']
            
            #Change Score, Position, Timeouts, Drive
            #Posteam Type
            if(gameDF['posteam_type'] == 'home'):
                if(gameDF['down'].iloc[0] == 3):
                    h3rdFail += 1
                elif(gameDF['down'].iloc[0] == 4):
                    h4thFail += 1
                gameDF['posteam_type'] = 'away'####################################################################################
                gameDF['totinterception'] = aInt
                gameDF['totincomplete_pass'] = aIncomplete
                gameDF['totcomplete_pass'] = aComplete
                gameDF['totThirdDown_convert'] = a3rdConvert
                gameDF['totThirdDown_fail'] = a3rdFail
                gameDF['totFourthDown_convert'] = a4thConvert
                gameDF['totFourthDown_fail'] = a4thFail
                gameDF['totPass_TD'] = aPassTD
                gameDF['totRush_TD'] = aRushTD
                gameDF['totfumble'] = aFumble
                gameDF['totHit'] = aHit
                gameDF['totfirst_down_rush'] = aFirstRush
                gameDF['totfirst_down_pass'] = aFirstPass
                gameDF['posteam_type'] = 'away'
                #Change teams (Punt)
                #Score
                temp = gameDF['posteam_score']
                gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
                gameDF['defteam_score'] = temp
                    
                #Possession
                temp = gameDF['posteam']
                gameDF['posteam'] = gameDF['defteam']
                gameDF['defteam'] = temp
                
                print("Change")
                print(gameDF['posteam'].iloc[0])
                print(gameDF['defteam'].iloc[0])
                print(temp)
                    
                #Timeouts
                temp = gameDF['posteam_timeouts_remaining'].iloc[0]
                gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
                gameDF['defteam_timeouts_remaining'] = temp
                
                gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                    
                #Drive
                gameDF['drive'] += 1#############################################################################################
            else:
                if(gameDF['down'].iloc[0] == 3):
                    a3rdFail += 1
                elif(gameDF['down'].iloc[0] == 4):
                    a4thFail += 1
                gameDF['posteam_type'] = 'home'###################################################################################
                gameDF['totinterception'] = hInt
                gameDF['totincomplete_pass'] = hIncomplete
                gameDF['totcomplete_pass'] = hComplete
                gameDF['totThirdDown_convert'] = h3rdConvert
                gameDF['totThirdDown_fail'] = h3rdFail
                gameDF['totFourthDown_convert'] = h4thConvert
                gameDF['totFourthDown_fail'] = h4thFail
                gameDF['totPass_TD'] = hPassTD
                gameDF['totRush_TD'] = hRushTD
                gameDF['totfumble'] = hFumble
                gameDF['totHit'] = hHit
                gameDF['totfirst_down_rush'] = hFirstRush
                gameDF['totfirst_down_pass'] = hFirstPass
                gameDF['posteam_type'] = 'home'
                #Score
                temp = gameDF['posteam_score'].iloc[0]
                gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
                gameDF['defteam_score'] = temp
                    
                #Possession
                temp = gameDF['posteam']
                gameDF['posteam'] = gameDF['defteam']
                gameDF['defteam'] = temp
                
                print("Change")
                print(gameDF['posteam'].iloc[0])
                print(gameDF['defteam'].iloc[0])
                print(temp)
                    
                #Timeouts
                temp = gameDF['posteam_timeouts_remaining'].iloc[0]
                gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
                gameDF['defteam_timeouts_remaining'] = temp
                
                gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                    
                #Drive
                gameDF['drive'] += 1#####################################################################################################
            
            gameDF['down'] = 1
            if(int(gameDF['yardline_100']) > 10):
                gameDF['ydstogo'] = 10
            else:
                gameDF['ydstogo'] = gameDF['yardline_100'].iloc[0]
            gameDF['duration'] = -25
        
        #NOT Intercepted
        else:
            print('Not Picked!')
            #Completed?
            gameDF['complete_pass'] = predCompletion(gameDF, gameDF['passer_player_name'].iloc[0], gameDF['pass_defense_1_player_name'].iloc[0], gameDF['receiver_player_name'].iloc[0])
            if(int(gameDF['complete_pass'].iloc[0]) == 1):
                print('Complete!')
                if(gameDF['posteam_type'].iloc[0] == 'home'):
                    hComplete += 1
                    gameDF['totincomplete_pass'] = hComplete
                else:
                    aComplete += 1
                    gameDF['totincomplete_pass'] = aComplete
                #Yards after Catch and Duration
                gameDF['yards_after_catch'] = predYardsAfterCatch(gameDF, gameDF['passer_player_name'].iloc[0], gameDF['pass_defense_1_player_name'].iloc[0], gameDF['receiver_player_name'].iloc[0])
                gameDF['duration'] = predPassDuration(gameDF, gameDF['receiver_player_name'].iloc[0])
                #Check First Down
                gameDF['ydstogo'] = gameDF['ydstogo'].iloc[0] - (gameDF['air_yards'].iloc[0]+gameDF['yards_after_catch'].iloc[0])
                gameDF['yardline_100'] = gameDF['yardline_100'].iloc[0] - (gameDF['air_yards'].iloc[0]+gameDF['yards_after_catch'].iloc[0])
                
                if(gameDF['ydstogo'].iloc[0] <= 0):
                    print('First Down!')
                    if(gameDF['posteam_type'].iloc[0] == 'home'):
                        if(gameDF['down'].iloc[0] == 3):
                            h3rdConvert += 1
                            gameDF['totThirdDown_convert'] = h3rdConvert
                        elif(gameDF['down'].iloc[0] == 4):
                            h4thConvert += 1
                            gameDF['totFourthDown_convert'] = h4thConvert
                    else:
                        if(gameDF['down'].iloc[0] == 3):
                            a3rdFail += 1
                            gameDF['totThirdDown_convert'] = a3rdConvert
                        elif(gameDF['down'].iloc[0] == 4):
                            a4thFail += 1
                            gameDF['totFourthDown_convert'] = a4thConvert
                    gameDF['down'] = 1
                    if(gameDF['yardline_100'].iloc[0] > 10):
                        gameDF['ydstogo'] = 10
                    else:
                        gameDF['ydstogo'] = gameDF['yardline_100'].iloc[0]
                    if(gameDF['posteam_type'].iloc[0] == 'home'):
                        hFirstPass += 1
                        gameDF['totfirst_down_pass'] = hFirstPass
                else:
                    if(gameDF['posteam_type'] == 'home'):
                        if(gameDF['down'].iloc[0] == 3):
                            h3rdFail += 1
                            gameDF['totThirdDown_fail'] = h3rdFail
                        elif(gameDF['down'].iloc[0] == 4):
                            h4thFail += 1
                            gameDF['totFourthDown_fail'] = h4thFail
                    else:
                        if(gameDF['down'].iloc[0] == 3):
                            a3rdFail += 1
                            gameDF['totThirdDown_fail'] = a3rdFail
                        elif(gameDF['down'].iloc[0] == 4):
                            a4thFail += 1
                            gameDF['totFourthDown_fail'] = a4thFail
                    gameDF['down'] += 1
                        
                        
            else:
                gameDF['duration'] = -25
                if(gameDF['posteam_type'].iloc[0] == 'home'):
                    hIncomplete += 1
                    gameDF['totincomplete_pass'] = hIncomplete
                    if(int(gameDF['down'].iloc[0]) == 3):
                        h3rdFail += 1
                        gameDF['totThirdDown_fail'] = h3rdFail
                    elif(int(gameDF['down'].iloc[0]) == 4):
                        h4thFail += 1
                        gameDF['totFourthDown_fail'] = h4thFail
                else:
                    aIncomplete += 1
                    gameDF['totincomplete_pass'] = aIncomplete
                    if(int(gameDF['down'].iloc[0]) == 3):
                        a3rdFail += 1
                        gameDF['totThirdDown_fail'] = a3rdFail
                    elif(int(gameDF['down'].iloc[0]) == 4):
                        a4thFail += 1
                        gameDF['totFourthDown_fail'] = a4thFail
                gameDF['down'] += 1
        
        if(int(gameDF['yardline_100'].iloc[0]) <= 0):
            justScored = 1
            #Add points for TD
            gameDF['posteam_score'] += 7
            gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
            if(gameDF['posteam_type'].iloc[0] == 'home'):
                gameDF['total_home_score'] += 7
            else:
                gameDF['total_away_score'] += 7
            gameDF['yardline_100'] = 75
            
        if(int(gameDF['down'].iloc[0]) == 5):
        
            #Change Score, Position, Timeouts, Drive
            #Posteam Type
            if(gameDF['posteam_type'].iloc[0] == 'home'):
                gameDF['posteam_type'] = 'away'####################################################################################
                gameDF['totinterception'] = aInt
                gameDF['totincomplete_pass'] = aIncomplete
                gameDF['totcomplete_pass'] = aComplete
                gameDF['totThirdDown_convert'] = a3rdConvert
                gameDF['totThirdDown_fail'] = a3rdFail
                gameDF['totFourthDown_convert'] = a4thConvert
                gameDF['totFourthDown_fail'] = a4thFail
                gameDF['totPass_TD'] = aPassTD
                gameDF['totRush_TD'] = aRushTD
                gameDF['totfumble'] = aFumble
                gameDF['totHit'] = aHit
                gameDF['totfirst_down_rush'] = aFirstRush
                gameDF['totfirst_down_pass'] = aFirstPass
                gameDF['posteam_type'] = 'away'
                #Change teams (Punt)
                #Score
                temp = gameDF['posteam_score'].iloc[0]
                gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
                gameDF['defteam_score'] = temp
                    
                #Possession
                temp = gameDF['posteam'].iloc[0]
                gameDF['posteam'] = gameDF['defteam'].iloc[0]
                gameDF['defteam'] = temp
                
                print("Change")
                print(gameDF['posteam'].iloc[0])
                print(gameDF['defteam'].iloc[0])
                print(temp)
                    
                #Timeouts
                temp = gameDF['posteam_timeouts_remaining'].iloc[0]
                gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
                gameDF['defteam_timeouts_remaining'] = temp
                
                gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                    
                #Drive
                gameDF['drive'] += 1#############################################################################################
            else:
                gameDF['posteam_type'] = 'home'###################################################################################
                gameDF['totinterception'] = hInt
                gameDF['totincomplete_pass'] = hIncomplete
                gameDF['totcomplete_pass'] = hComplete
                gameDF['totThirdDown_convert'] = h3rdConvert
                gameDF['totThirdDown_fail'] = h3rdFail
                gameDF['totFourthDown_convert'] = h4thConvert
                gameDF['totFourthDown_fail'] = h4thFail
                gameDF['totPass_TD'] = hPassTD
                gameDF['totRush_TD'] = hRushTD
                gameDF['totfumble'] = hFumble
                gameDF['totHit'] = hHit
                gameDF['totfirst_down_rush'] = hFirstRush
                gameDF['totfirst_down_pass'] = hFirstPass
                gameDF['posteam_type'] = 'home'
                #Score
                temp = gameDF['posteam_score'].iloc[0]
                gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
                gameDF['defteam_score'] = temp
                    
                #Possession
                temp = gameDF['posteam'].iloc[0]
                gameDF['posteam'] = gameDF['defteam'].iloc[0]
                gameDF['defteam'] = temp
                
                print("Change")
                print(gameDF['posteam'].iloc[0])
                print(gameDF['defteam'].iloc[0])
                print(temp)
                    
                #Timeouts
                temp = gameDF['posteam_timeouts_remaining'].iloc[0]
                gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
                gameDF['defteam_timeouts_remaining'] = temp
                
                gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                    
                #Drive
                gameDF['drive'] += 1#####################################################################################################
            #Reset Stuff
            gameDF['ydstogo'] = 10
            gameDF['down'] = 1
        #Take off time
        gameDF['game_seconds_remaining'] += gameDF['duration'].iloc[0]
        gameDF['half_seconds_remaining'] += gameDF['duration'].iloc[0]
        gameDF['quarter_seconds_remaining'] += gameDF['duration'].iloc[0]
        if(int(gameDF['quarter_seconds_remaining'].iloc[0]) <= 0):
            gameDF['qtr'] += 1
            gameDF['quarter_seconds_remaining'] = 900
        if(int(gameDF['half_seconds_remaining'].iloc[0]) <= 0):
            gameDF['half_seconds_remaining'] = 1800
            gameDF['game_seconds_remaining'] = 1800
    #################################################################################################################################
    #
    #   RUN
    #
    #################################################################################################################################
    else:
        rushes += 1
        passes = 0
        print('RUN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(gameDF['pass_attempt'].iloc[0])
        gameDF['rusher_player_name'] = predRushPlayer(gameDF)
        gameDF['run_location'] = predRunLocation(gameDF, gameDF['rusher_player_name'].iloc[0])
        gameDF['run_gap'] = predRunGap(gameDF, gameDF['rusher_player_name'].iloc[0])
        gameDF['yards_gained'] = predRushYards(gameDF, gameDF['rusher_player_name'].iloc[0])
        #gameDF['duration'] = predRunDuration(gameDF, gameDF['rusher_player_name'].iloc[0])
        gameDF['duration'] = -30
        #Check First Down
        gameDF['ydstogo'] = gameDF['ydstogo'].iloc[0] - (gameDF['yards_gained'].iloc[0])
        if(gameDF['ydstogo'].iloc[0] <= 0):
            if(gameDF['posteam_type'].iloc[0] == 'home'):
                if(gameDF['down'].iloc[0] == 3):
                    h3rdConvert += 1
                    gameDF['totThirdDown_convert'] = h3rdConvert
                elif(gameDF['down'].iloc[0] == 4):
                    h4thConvert += 1
                    gameDF['totFourthDown_convert'] = h4thConvert
            else:
                if(gameDF['down'].iloc[0] == 3):
                    a3rdFail += 1
                    gameDF['totThirdDown_convert'] = a3rdConvert
                elif(gameDF['down'].iloc[0] == 4):
                    a4thFail += 1
                    gameDF['totFourthDown_convert'] = a4thConvert
            gameDF['down'] = 1
            if(gameDF['yardline_100'].iloc[0] > 10):
                gameDF['ydstogo'] = 10
            if(gameDF['posteam_type'].iloc[0] == 'home'):
                hFirstRush += 1
                gameDF['totfirst_down_pass'] = hFirstRush
            else:
                aFirstRush += 1
                gameDF['totfirst_down_pass'] = aFirstRush
        else:
            if(gameDF['posteam_type'].iloc[0] == 'home'):
                if(gameDF['down'].iloc[0] == 3):
                    h3rdFail += 1
                    gameDF['totThirdDown_fail'] = h3rdFail
                elif(gameDF['down'].iloc[0] == 4):
                    h4thFail += 1
                    gameDF['totFourthDown_fail'] = h4thFail
            else:
                if(gameDF['down'].iloc[0] == 3):
                    a3rdFail += 1
                    gameDF['totThirdDown_fail'] = a3rdFail
                elif(gameDF['down'].iloc[0] == 4):
                    a4thFail += 1
                    gameDF['totFourthDown_fail'] = a4thFail
            gameDF['down'] += 1
            
        gameDF['yardline_100'] = gameDF['yardline_100'].iloc[0]-gameDF['yards_gained'].iloc[0]
        if(gameDF['yardline_100'].iloc[0] <= 0):
            justScored = 1
            #Add points for TD
            gameDF['posteam_score'] += 7
            gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
            if(gameDF['posteam_type'].loc[0] == 'home'):
                gameDF['total_home_score'] += 7
            else:
                gameDF['total_away_score'] += 7
            gameDF['yardline_100'] = 75
            
            #Change Score, Position, Timeouts, Drive
            #Posteam Type
            if(gameDF['posteam_type'].iloc[0] == 'home'):
                gameDF['posteam_type'] = 'away'####################################################################################
                gameDF['totinterception'] = aInt
                gameDF['totincomplete_pass'] = aIncomplete
                gameDF['totcomplete_pass'] = aComplete
                gameDF['totThirdDown_convert'] = a3rdConvert
                gameDF['totThirdDown_fail'] = a3rdFail
                gameDF['totFourthDown_convert'] = a4thConvert
                gameDF['totFourthDown_fail'] = a4thFail
                gameDF['totPass_TD'] = aPassTD
                gameDF['totRush_TD'] = aRushTD
                gameDF['totfumble'] = aFumble
                gameDF['totHit'] = aHit
                gameDF['totfirst_down_rush'] = aFirstRush
                gameDF['totfirst_down_pass'] = aFirstPass
                gameDF['posteam_type'] = 'away'
                #Change teams (Punt)
                #Score
                temp = gameDF['posteam_score'].iloc[0]
                gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
                gameDF['defteam_score'] = temp
                    
                #Possession
                temp = gameDF['posteam'].iloc[0]
                gameDF['posteam'] = gameDF['defteam'].iloc[0]
                gameDF['defteam'] = temp
                print("Change")
                print(gameDF['posteam'].iloc[0])
                print(gameDF['defteam'].iloc[0])
                print(temp)
                    
                #Timeouts
                temp = gameDF['posteam_timeouts_remaining'].iloc[0]
                gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
                gameDF['defteam_timeouts_remaining'] = temp
                
                gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                    
                #Drive
                gameDF['drive'] += 1#############################################################################################
            else:
                gameDF['posteam_type'] = 'home'###################################################################################
                gameDF['totinterception'] = hInt
                gameDF['totincomplete_pass'] = hIncomplete
                gameDF['totcomplete_pass'] = hComplete
                gameDF['totThirdDown_convert'] = h3rdConvert
                gameDF['totThirdDown_fail'] = h3rdFail
                gameDF['totFourthDown_convert'] = h4thConvert
                gameDF['totFourthDown_fail'] = h4thFail
                gameDF['totPass_TD'] = hPassTD
                gameDF['totRush_TD'] = hRushTD
                gameDF['totfumble'] = hFumble
                gameDF['totHit'] = hHit
                gameDF['totfirst_down_rush'] = hFirstRush
                gameDF['totfirst_down_pass'] = hFirstPass
                gameDF['posteam_type'] = 'home'
                #Score
                temp = gameDF['posteam_score'].iloc[0]
                gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
                gameDF['defteam_score'] = temp
                    
                #Possession
                temp = gameDF['posteam'].iloc[0]
                gameDF['posteam'] = gameDF['defteam'].iloc[0]
                gameDF['defteam'] = temp
                
                print("Change")
                print(gameDF['posteam'].iloc[0])
                print(gameDF['defteam'].iloc[0])
                print(temp)
                    
                #Timeouts
                temp = gameDF['posteam_timeouts_remaining'].iloc[0]
                gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
                gameDF['defteam_timeouts_remaining'] = temp
                
                gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                    
                #Drive
                gameDF['drive'] += 1#####################################################################################################
            #Reset Stuff
            gameDF['ydstogo'] = 10
            gameDF['down'] = 1
            
            
        #Take off time
        gameDF['game_seconds_remaining'] += gameDF['duration'].iloc[0]
        gameDF['half_seconds_remaining'] += gameDF['duration'].iloc[0]
        gameDF['quarter_seconds_remaining'] += gameDF['duration'].iloc[0]
        if(gameDF['quarter_seconds_remaining'].iloc[0] <= 0):
            gameDF['qtr'] += 1
            gameDF['quarter_seconds_remaining'] = 900
        if(gameDF['half_seconds_remaining'].iloc[0] <= 0):
            gameDF['half_seconds_remaining'] = 1800
            gameDF['game_seconds_remaining'] = 1800
    #################################################################################################################################
    #
    #   TURNOVER ON DOWNS
    #
    #################################################################################################################################
    if(gameDF['down'].iloc[0] == 5):
        gameDF['yardline_100'] = 100 - gameDF['yardline_100']
        
        #Change Score, Position, Timeouts, Drive
        #Posteam Type
        if(gameDF['posteam_type'].iloc[0] == 'home'):
            gameDF['posteam_type'] = 'away'####################################################################################
            gameDF['totinterception'] = aInt
            gameDF['totincomplete_pass'] = aIncomplete
            gameDF['totcomplete_pass'] = aComplete
            gameDF['totThirdDown_convert'] = a3rdConvert
            gameDF['totThirdDown_fail'] = a3rdFail
            gameDF['totFourthDown_convert'] = a4thConvert
            gameDF['totFourthDown_fail'] = a4thFail
            gameDF['totPass_TD'] = aPassTD
            gameDF['totRush_TD'] = aRushTD
            gameDF['totfumble'] = aFumble
            gameDF['totHit'] = aHit
            gameDF['totfirst_down_rush'] = aFirstRush
            gameDF['totfirst_down_pass'] = aFirstPass
            gameDF['posteam_type'] = 'away'
            #Change teams (Punt)
            #Score
            temp = gameDF['posteam_score'].iloc[0]
            gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
            gameDF['defteam_score'] = temp
                
            #Possession
            temp = gameDF['posteam'].iloc[0]
            gameDF['posteam'] = gameDF['defteam'].iloc[0]
            gameDF['defteam'] = temp
            
            print("Change")
            print(gameDF['posteam'].iloc[0])
            print(gameDF['defteam'].iloc[0])
            print(temp)
                
            #Timeouts
            temp = gameDF['posteam_timeouts_remaining'].iloc[0]
            gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
            gameDF['defteam_timeouts_remaining'] = temp
            
            gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                
            #Drive
            gameDF['drive'] += 1#############################################################################################
        else:
            gameDF['posteam_type'] = 'home'###################################################################################
            gameDF['totinterception'] = hInt
            gameDF['totincomplete_pass'] = hIncomplete
            gameDF['totcomplete_pass'] = hComplete
            gameDF['totThirdDown_convert'] = h3rdConvert
            gameDF['totThirdDown_fail'] = h3rdFail
            gameDF['totFourthDown_convert'] = h4thConvert
            gameDF['totFourthDown_fail'] = h4thFail
            gameDF['totPass_TD'] = hPassTD
            gameDF['totRush_TD'] = hRushTD
            gameDF['totfumble'] = hFumble
            gameDF['totHit'] = hHit
            gameDF['totfirst_down_rush'] = hFirstRush
            gameDF['totfirst_down_pass'] = hFirstPass
            gameDF['posteam_type'] = 'home'
            #Score
            temp = gameDF['posteam_score'].iloc[0]
            gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
            gameDF['defteam_score'] = temp
                
            #Possession
            temp = gameDF['posteam'].iloc[0]
            gameDF['posteam'] = gameDF['defteam'].iloc[0]
            gameDF['defteam'] = temp
            
            print("Change")
            print(gameDF['posteam'].iloc[0])
            print(gameDF['defteam'].iloc[0])
            print(temp)
                
            #Timeouts
            temp = gameDF['posteam_timeouts_remaining'].iloc[0]
            gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
            gameDF['defteam_timeouts_remaining'] = temp
            
            gameDF['score_differential'] = gameDF['defteam_score'] - gameDF['posteam_score'].iloc[0]
                
            #Drive
            gameDF['drive'] += 1#####################################################################################################
    ####################################################################################################################################
    #   Check if Just Scored a TD by Throw OR Run
    ####################################################################################################################################
    if(justScored == 1):
        gameDF['yardline_100'] = 75
        
        #Change Score, Position, Timeouts, Drive
        #Posteam Type
        if(gameDF['posteam_type'].iloc[0] == 'home'):
            gameDF['posteam_type'] = 'away'####################################################################################
            gameDF['totinterception'] = aInt
            gameDF['totincomplete_pass'] = aIncomplete
            gameDF['totcomplete_pass'] = aComplete
            gameDF['totThirdDown_convert'] = a3rdConvert
            gameDF['totThirdDown_fail'] = a3rdFail
            gameDF['totFourthDown_convert'] = a4thConvert
            gameDF['totFourthDown_fail'] = a4thFail
            gameDF['totPass_TD'] = aPassTD
            gameDF['totRush_TD'] = aRushTD
            gameDF['totfumble'] = aFumble
            gameDF['totHit'] = aHit
            gameDF['totfirst_down_rush'] = aFirstRush
            gameDF['totfirst_down_pass'] = aFirstPass
            gameDF['posteam_type'] = 'away'
            #Change teams (Punt)
            #Score
            temp = gameDF['posteam_score'].iloc[0]
            gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
            gameDF['defteam_score'] = temp
                
            #Possession
            temp = gameDF['posteam'].iloc[0]
            gameDF['posteam'] = gameDF['defteam'].iloc[0]
            gameDF['defteam'] = temp
            
            print("Change")
            print(gameDF['posteam'].iloc[0])
            print(gameDF['defteam'].iloc[0])
            print(temp)
                
            #Timeouts
            temp = gameDF['posteam_timeouts_remaining'].iloc[0]
            gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
            gameDF['defteam_timeouts_remaining'] = temp
            
            gameDF['score_differential'] = gameDF['defteam_score'].iloc[0] - gameDF['posteam_score'].iloc[0]
                
            #Drive
            gameDF['drive'] += 1#############################################################################################
        else:
            gameDF['posteam_type'] = 'home'###################################################################################
            gameDF['totinterception'] = hInt
            gameDF['totincomplete_pass'] = hIncomplete
            gameDF['totcomplete_pass'] = hComplete
            gameDF['totThirdDown_convert'] = h3rdConvert
            gameDF['totThirdDown_fail'] = h3rdFail
            gameDF['totFourthDown_convert'] = h4thConvert
            gameDF['totFourthDown_fail'] = h4thFail
            gameDF['totPass_TD'] = hPassTD
            gameDF['totRush_TD'] = hRushTD
            gameDF['totfumble'] = hFumble
            gameDF['totHit'] = hHit
            gameDF['totfirst_down_rush'] = hFirstRush
            gameDF['totfirst_down_pass'] = hFirstPass
            gameDF['posteam_type'] = 'home'
            #Score
            temp = gameDF['posteam_score'].iloc[0]
            gameDF['posteam_score'] = gameDF['defteam_score'].iloc[0]
            gameDF['defteam_score'] = temp
                
            #Possession
            temp = gameDF['posteam'].iloc[0]
            gameDF['posteam'] = gameDF['defteam'].iloc[0]
            gameDF['defteam'] = temp
            
            print("Change")
            print(gameDF['posteam'].iloc[0])
            print(gameDF['defteam'].iloc[0])
            print(temp)
                
            #Timeouts
            temp = gameDF['posteam_timeouts_remaining'].iloc[0]
            gameDF['posteam_timeouts_remaining'] = gameDF['defteam_timeouts_remaining'].iloc[0]
            gameDF['defteam_timeouts_remaining'] = temp
            
            gameDF['score_differential'] = gameDF['defteam_score'] - gameDF['posteam_score'].iloc[0]
                
            #Drive
            gameDF['drive'] += 1#####################################################################################################


    
    print('POSTEAM')
    print(gameDF['posteam'].iloc[0])
    print('DOWN')
    print(gameDF['down'].iloc[0])
    print('DISTANCE')
    print(gameDF['ydstogo'].iloc[0])
    print('HOME SCORE')
    print(gameDF['total_home_score'].iloc[0])
    print('AWAY SCORE')
    print(gameDF['total_away_score'].iloc[0])
    print('QTR')
    print(gameDF['qtr'].iloc[0])
    print('TIME')
    print(gameDF['quarter_seconds_remaining'].iloc[0])
    print('YARDS GAINED')
    print(gameDF['yards_gained'].iloc[0])
    print('100 Yard Line')
    print(gameDF['yardline_100'].iloc[0])
    print('Duration')
    print(gameDF['duration'].iloc[0])
    print('Air Yards')
    print(gameDF['air_yards'].iloc[0])
    print('Yards After Catch')
    print(gameDF['yards_after_catch'].iloc[0])
    gameDF.to_csv('gameFinal.csv', index=False)

gameDF.to_csv('gameFinal.csv', index=False)