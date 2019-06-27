from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc
import numpy as np  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

tf.enable_eager_execution()

#trainTimeOut()
#trainKnee()
#trainSpike()
#trainml()
#trainRunLocation()
#trainRunGap()
#trainRunYards()
#trainQBHit()
#trainSack()
#trainPassLocation()
#trainIncompletePass()
#trainAirYards()
#trainInterception()
#trainPassYards()
#trainFumble()
#trainFumbleLost()
#trainDuration()
#trainPos() 
#trainFieldGoal()

def fieldGoalclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def fieldGoalthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_gap':
            train_stats = train_stats.drop(c, axis=1)   
        #elif c == 'field_goal_result':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainFieldGoal():
    df = pd.read_csv('fieldGoal.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #df['try'] = 0
    #df = df.drop(columns=['extra_point_attempt', 'two_point_attempt'])


    #Change String stats to dummy columns
    df = fieldGoalclean(df)
    #Throw Out stats that are 'illegal'
    df = fieldGoalthrowOut(df)

    df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('field_goal_result', axis=1)
    y = df['field_goal_result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)

def posclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def posthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        #elif c == 'air_yards':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'pass_location':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_location':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'kick_distance':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'timeout':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'incomplete_pass':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'interception':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'safety':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'fumble_lost':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'qb_hit':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'sack':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'fumble':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'complete_pass':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'fumble_recovery_1_yards':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'return_yards':
        #   train_stats = train_stats.drop(c, axis=1)
        #elif c == 'duration':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_gap':
        #    train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'Index':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats

def trainPos(week, df):
    if 1 == 1:
        #data = pd.read_csv('testLast2.csv', low_memory=False)
        df = df.loc[(df['posteam'] != '0')]
        df = df.loc[(df['Year'] == 2018) & (df['SType'] == 'Regular') & (df['Week'] <= week)]
        #QB = data['passer_player_id'].unique()
        #RB = data['rusher_player_id'].unique()
        #WR = data['receiver_player_id'].unique()
        #K = data['kicker_player_id'].unique()
        
        

        #df = df.drop(columns=['passer_player_id_x', 'receiver_player_id_x', 'rusher_player_id_x', 'kicker_player_id_x',
        #                'passer_player_id_y', 'receiver_player_id_y', 'rusher_player_id_y', 'kicker_player_id_y'])

        df['passLocation'] = 0
        df.loc[df.pass_location == 'left', 'passLocation'] = 1
        df.loc[df.pass_location == 'middle', 'passLocation'] = 2
        df.loc[df.pass_location == 'right', 'passLocation'] = 3
        df = df.drop(columns=['pass_location'])

        df['gap'] = 0
        df.loc[df.run_gap == 'end', 'gap'] = 1
        df.loc[df.run_gap == 'tackle', 'gap'] = 2
        df.loc[df.run_gap == 'gaurd', 'gap'] = 3
        df = df.drop(columns=['run_gap'])

        df['location'] = 0
        df.loc[df.run_location == 'left', 'location'] = 1
        df.loc[df.run_location == 'middle', 'location'] = 2
        df.loc[df.run_location == 'right', 'location'] = 3
        df = df.drop(columns=['run_location'])
        
        
        
        
        
        
        QBtrans = np.loadtxt('QB.txt', dtype='str')
        RBtrans = np.loadtxt('RB.txt', dtype='str')
        WRtrans = np.loadtxt('WR.txt', dtype='str')
        Ktrans = np.loadtxt('K.txt', dtype='str')
        
        QBdata = data.loc[data['passer_player_id'] != '0']
        RBdata = data.loc[data['rusher_player_id'] != '0']
        WRdata = data.loc[data['receiver_player_id'] != '0']
        Kdata = data.loc[data['kicker_player_id'] != '0']
        
        QBdf = df.loc[df['pass_attempt'].astype(int) != 0]
        RBdf = df.loc[df['rush_attempt'].astype(int) != 0]
        WRdf = df.loc[df['pass_attempt'].astype(int) != 0]
        Kdf = df.loc[df['field_goal_attempt'].astype(int) != 0]
        
        
        
        ####################################################################################################################
        #   QUARTERBACKS
        ####################################################################################################################
        
        for player in range(len(QBtrans)):
            QBdata.loc[QBdata.passer_player_id == QBtrans[player], 'passer_player_id'] = player
        for pindex, prow in QBdf.iterrows():
            for did, drow in QBdata.iterrows():
                if(QBdf['passer_player_id'][pindex] == QBdata['passer_player_id'][did]):
                    QBdf['passer_player_id'][pindex] = QBdata['passer_player_id'][did]


        teams = QBdf['posteam'].unique()

        #Change String stats to dummy columns
        QBdf = posclean(QBdf)
        #Throw Out stats that are 'illegal'
        QBdf = posthrowOut(QBdf)

        X = QBdf.drop('passer_player_id', axis=1)
        y = QBdf['passer_player_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        qbrf = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

        qbrf.fit(X_train, y_train)

        y_predict = qbrf.predict(X_test)
        print("Accuracy")
        ab = accuracy_score(y_test, y_predict)
         
        print(ab) 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ####################################################################################################################
        #   RUNNING BACKS
        ####################################################################################################################
        
        for player in range(len(RBtrans)):
            RBdata.loc[RBdata.rusher_player_id == RBtrans[player], 'rusher_player_id'] = player
        for pindex, prow in RBdf.iterrows():
            for did, drow in RBdata.iterrows():
                if(RBdf['rusher_player_id'][pindex] == RBdata['rusher_player_id'][did]):
                    RBdf['rusher_player_id'][pindex] = RBdata['rusher_player_id'][did]


        teams = RBdf['posteam'].unique()

        #Change String stats to dummy columns
        RBdf = posclean(RBdf)
        #Throw Out stats that are 'illegal'
        RBdf = posthrowOut(RBdf)

        X = RBdf.drop('rusher_player_id', axis=1)
        y = RBdf['rusher_player_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        rbrf = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

        rbrf.fit(X_train, y_train)

        y_predict = rbrf.predict(X_test)
        print("Accuracy")
        ab = accuracy_score(y_test, y_predict)
         
        print(ab)  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ####################################################################################################################
        #    WIDE RECEIVERS
        ####################################################################################################################
        
        for player in range(len(WRtrans)):
            WRdata.loc[WRdata.receiver_player_id == WRtrans[player], 'receiver_player_id'] = player
        for pindex, prow in WRdf.iterrows():
            for did, drow in WRdata.iterrows():
                if(WRdf['receiver_player_id'][pindex] == WRdata['receiver_player_id'][did]):
                    WRdf['receiver_player_id'][pindex] = WRdata['receiver_player_id'][did]


        teams = WRdf['posteam'].unique()

        #Change String stats to dummy columns
        WRdf = posclean(WRdf)
        #Throw Out stats that are 'illegal'
        WRdf = posthrowOut(WRdf)

        X = WRdf.drop('receiver_player_id', axis=1)
        y = WRdf['receiver_player_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        wrdf = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

        wrdf.fit(X_train, y_train)

        y_predict = wrdf.predict(X_test)
        print("Accuracy")
        ab = accuracy_score(y_test, y_predict)
         
        print(ab)  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ####################################################################################################################
        #   KICKERS
        ####################################################################################################################
        if 'Ktrans' in globals():
            for player in range(len(Ktrans)):
                Kdata.loc[Kdata.kicker_player_id == Ktrans[player], 'kicker_player_id'] = player
            for pindex, prow in Kdf.iterrows():
                for did, drow in Kdata.iterrows():
                    if(Kdf['kicker_player_id'][pindex] == Kdata['kicker_player_id'][did]):
                        Kdf['kicker_player_id'][pindex] = Kdata['kicker_player_id'][did]


        teams = Kdf['posteam'].unique()

        #Change String stats to dummy columns
        Kdf = posclean(Kdf)
        #Throw Out stats that are 'illegal'
        Kdf = posthrowOut(Kdf)

        X = Kdf.drop('kicker_player_id', axis=1)
        y = Kdf['kicker_player_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        krf = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

        krf.fit(X_train, y_train)

        y_predict = krf.predict(X_test)
        print("Accuracy")
        ab = accuracy_score(y_test, y_predict)
         
        print(ab)  

    return(qbrf, rbrf, wrrf, krd)

def durationclean(training_df):
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
    training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
    training_df['extra_point_result'] = 'Extra' + training_df['extra_point_result'].astype(str)
    training_df['two_point_conv_result'] = 'Two' + training_df['two_point_conv_result'].astype(str)
    training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
    training_df['field_goal_result'] = 'Field' + training_df['field_goal_result'].astype(str)
    training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
    #training_df['td_team'] = 'TD' + training_df['td_team'].astype(str)
    training_df['passer_player_id'] = 'Pass2' + training_df['passer_player_id'].astype(str)
    training_df['receiver_player_id'] = 'Rec' + training_df['receiver_player_id'].astype(str)
    training_df['rusher_player_id'] = 'Rush' + training_df['rusher_player_id'].astype(str)
    training_df['kicker_player_id'] = 'Kick' + training_df['kicker_player_id'].astype(str)
    #training_df['penalty_team'] = 'PTeam' + training_df['penalty_team'].astype(str)
    training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
    training_df['WDirection'] = training_df['WDirection'].astype(str)
    training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
    training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
    training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
    training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
    training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    #training_df['fumbled_1_player_id'] = 'fum' + training_df['fumbled_1_player_id'].astype(str)

    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    Aoff = list(pd.get_dummies(training_df['AOffense']).columns.values)
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['penalty_team'])], axis=1)
    #training_df = training_df.drop(columns=['penalty_team'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['kicker_player_id'])], axis=1)
    training_df = training_df.drop(columns=['kicker_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['rusher_player_id'])], axis=1)
    training_df = training_df.drop(columns=['rusher_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['receiver_player_id'])], axis=1)
    training_df = training_df.drop(columns=['receiver_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['passer_player_id'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['td_team'])], axis=1)
    #training_df = training_df.drop(columns=['td_team'])
    timeOuts = list(pd.get_dummies(training_df['timeout_team']).columns.values)#############################################3
    training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)
    #training_df = training_df.drop(columns=['timeout_team'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['two_point_conv_result'])], axis=1)
    training_df = training_df.drop(columns=['two_point_conv_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['extra_point_result'])], axis=1)
    training_df = training_df.drop(columns=['extra_point_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['field_goal_result'])], axis=1)
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_gap'])], axis=1)
    training_df = training_df.drop(columns=['run_gap'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_location'])], axis=1)
    training_df = training_df.drop(columns=['pass_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam_type'])], axis=1)
    training_df = training_df.drop(columns=['posteam_type'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['fumbled_1_player_id'])], axis=1)
    training_df = training_df.drop(columns=['fumbled_1_player_id'])
    #training_df = training_df.drop(columns=['goal_to_go'])
    return training_df

def durationthrowOut(train_stats):
    for c in train_stats.columns.values.astype(str):
        #print(c)
        #print(type(c))
        #if c == '0':
        #    train_stats = train_stats.drop(c, axis=1)
        if c == 'yards_gained':
            train_stats = train_stats.drop(c, axis=1)
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Pass'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'air_yards':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'yards_after_catch':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Run'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Gap'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Field'):
        #   train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Extra'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Two'):
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('TO'):
        #    train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('TD'):
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'field_goal_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'punt_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Pass2'):
           train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Rec'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Rush'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Kick'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('fum'):
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        return train_stats

def trainDuration():
    print(tf.__version__)
    training_df: pd.DataFrame = pd.read_csv("duration.csv", low_memory=False)
    training_df.loc[training_df.WTemp == '39/53', 'WTemp'] = 46   
    
    training_df = durationclean(training_df)
    training_df = durationthrowOut(training_df)
    training_df = training_df.loc[training_df.pass_attempt == 1]
    training_df = training_df.loc[training_df.incomplete_pass == 0]

    training_df = training_df.astype(float)
    dataset = training_df.copy()
    del training_df
    print(dataset.tail())
    gc.collect()
    print(dataset)
    dataset = dataset.astype(float)

    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    del dataset
    gc.collect()
    print("good")
    print("gooder")
    train_stats = train_dataset.describe()
    train_stats.pop("duration")
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset.pop('duration')
    test_labels = test_dataset.pop('duration')
    #train_labels = throwOut(train_labels)   
    #test_labels = throwOut(test_labels)   

    #train_labels = train_dataset.pop(Aoff)

    #def norm(x):
        #return (x - x['yards_gained'].sum()/x['yards_gained'].count() / train_stats['yards_gained'].std()

    #normed_train_data = norm(train_dataset)
    #normed_test_data = norm(test_dataset)
    print('Good')

    def build_model():
        model = keras.Sequential([
          layers.Dense(1600, activation='relu', input_shape=[len(train_dataset.keys())]),
          layers.Dense(1000, activation='relu'),
          layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model
        
    model = build_model()

    model.summary()

    example_batch = train_dataset[:10]
    example_result = model.predict(example_batch)

    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 10 == 0: 
                print('')
            print('.', end='')

    EPOCHS = 50

    history = model.fit(
        train_dataset, train_labels,
        epochs=EPOCHS, validation_split = 0.2, verbose=0,
        callbacks=[PrintDot()])
        
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [duration]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label = 'Val Error')
        plt.ylim([-100,1000])
        plt.legend()
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$duration^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label = 'Val Error')
        plt.ylim([-100,2000])
        plt.legend()
        plt.show()


    plot_history(history)

    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)

    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    test_predictions = model.predict(test_dataset).flatten()
     
    plt.scatter(test_labels, test_predictions) 
    plt.xlabel('True Values [duration]')
    plt.ylabel('Predictions [duration]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([-100,plt.xlim()[1]])
    plt.ylim([-100,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [duration]")
    _ = plt.ylabel("Count")
    return(model)

def fumbleLostclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
    training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
    training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
    training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])
    
    training_df = training_df.drop(columns=['run_gap'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_location'])], axis=1)
    training_df = training_df.drop(columns=['pass_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    timeOuts = list(pd.get_dummies(training_df['timeout_team']).columns.values)#############################################3
    training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)

    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    return training_df


def fumbleLostthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        #elif c == 'air_yards':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'pass_location':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_location':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'kick_distance':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'timeout':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'fumble_lost':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'qb_hit':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'sack':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'fumble':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_gap':
        #    train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainFumbleLost():
    df = pd.read_csv('fumbleLost.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = fumbleLostclean(df)
    #Throw Out stats that are 'illegal'
    df = fumbleLostthrowOut(df)

    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('fumble_lost', axis=1)
    y = df['fumble_lost']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)

def fumbleclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
    training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
    training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
    training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])
    
    training_df = training_df.drop(columns=['run_gap'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_location'])], axis=1)
    training_df = training_df.drop(columns=['pass_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    timeOuts = list(pd.get_dummies(training_df['timeout_team']).columns.values)#############################################3
    training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)

    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    return training_df


def fumblethrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        #elif c == 'air_yards':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'pass_location':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_location':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'kick_distance':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'timeout':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'qb_hit':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'sack':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'fumble':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_gap':
        #    train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainFumble():
    df = pd.read_csv('fumble.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = fumbleclean(df)
    #Throw Out stats that are 'illegal'
    df = fumblethrowOut(df)

    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('fumble', axis=1)
    y = df['fumble']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)

def passYardsclean(training_df):
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
    training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
    training_df['extra_point_result'] = 'Extra' + training_df['extra_point_result'].astype(str)
    training_df['two_point_conv_result'] = 'Two' + training_df['two_point_conv_result'].astype(str)
    training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
    training_df['field_goal_result'] = 'Field' + training_df['field_goal_result'].astype(str)
    training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
    #training_df['td_team'] = 'TD' + training_df['td_team'].astype(str)
    training_df['passer_player_id'] = 'Pass2' + training_df['passer_player_id'].astype(str)
    training_df['receiver_player_id'] = 'Rec' + training_df['receiver_player_id'].astype(str)
    training_df['rusher_player_id'] = 'Rush' + training_df['rusher_player_id'].astype(str)
    training_df['kicker_player_id'] = 'Kick' + training_df['kicker_player_id'].astype(str)
    #training_df['penalty_team'] = 'PTeam' + training_df['penalty_team'].astype(str)
    training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
    training_df['WDirection'] = training_df['WDirection'].astype(str)
    training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
    training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
    training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
    training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
    training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    #training_df['fumbled_1_player_id'] = 'fum' + training_df['fumbled_1_player_id'].astype(str)

    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    Aoff = list(pd.get_dummies(training_df['AOffense']).columns.values)
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['penalty_team'])], axis=1)
    #training_df = training_df.drop(columns=['penalty_team'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['kicker_player_id'])], axis=1)
    training_df = training_df.drop(columns=['kicker_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['rusher_player_id'])], axis=1)
    training_df = training_df.drop(columns=['rusher_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['receiver_player_id'])], axis=1)
    training_df = training_df.drop(columns=['receiver_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['passer_player_id'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['td_team'])], axis=1)
    #training_df = training_df.drop(columns=['td_team'])
    timeOuts = list(pd.get_dummies(training_df['timeout_team']).columns.values)#############################################3
    training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)
    training_df = training_df.drop(columns=['timeout_team'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['two_point_conv_result'])], axis=1)
    training_df = training_df.drop(columns=['two_point_conv_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['extra_point_result'])], axis=1)
    training_df = training_df.drop(columns=['extra_point_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['field_goal_result'])], axis=1)
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_gap'])], axis=1)
    training_df = training_df.drop(columns=['run_gap'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_location'])], axis=1)
    training_df = training_df.drop(columns=['pass_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam_type'])], axis=1)
    training_df = training_df.drop(columns=['posteam_type'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['fumbled_1_player_id'])], axis=1)
    training_df = training_df.drop(columns=['fumbled_1_player_id'])
    #training_df = training_df.drop(columns=['goal_to_go'])
    return training_df


def passYardsthrowOut(train_stats):
    for c in train_stats.columns.values.astype(str):
        #print(c)
        #print(type(c))
        #if c == '0':
        #    train_stats = train_stats.drop(c, axis=1)
        #if c == 'yards_gained':
        #    train_stats = train_stats.drop(c, axis=1)
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Pass'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'air_yards':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Run'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Gap'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Field'):
        #   train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Extra'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Two'):
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('TO'):
        #    train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('TD'):
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'field_goal_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'punt_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Pass2'):
           train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Rec'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Rush'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Kick'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('fum'):
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        return train_stats


def trainPassYards(df):
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = passYardsclean(df)
    #Throw Out stats that are 'illegal'
    df = passYardsthrowOut(df)


    #Numeric Value for each play type
    #Pass       0
    #Runs       1
    #Field Goal 2
    #Punt       3
    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('yards_gained', axis=1)
    y = df['yards_gained']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  

    regressor = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=0)  
    regressor.fit(X_train, y_train)  
    y_pred = regressor.predict(X_test)  

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    return(regressor)

def interceptionclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def interceptionthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        #elif c == 'air_yards':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'pass_location':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'incomplete_pass':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'interception':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_gap':
            train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainInterception():
    df = pd.read_csv('runLocation.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    df['passLocation'] = 0
    df.loc[df.pass_location == 'left', 'passLocation'] = 1
    df.loc[df.pass_location == 'middle', 'passLocation'] = 2
    df.loc[df.pass_location == 'right', 'passLocation'] = 3
    df = df.drop(columns=['pass_location'])

    #Change String stats to dummy columns
    df = interceptionclean(df)
    #Throw Out stats that are 'illegal'
    df = interceptionthrowOut(df)

    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('interception', axis=1)
    y = df['interception']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)

def airYardsclean(training_df):
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
    training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
    training_df['extra_point_result'] = 'Extra' + training_df['extra_point_result'].astype(str)
    training_df['two_point_conv_result'] = 'Two' + training_df['two_point_conv_result'].astype(str)
    training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
    training_df['field_goal_result'] = 'Field' + training_df['field_goal_result'].astype(str)
    training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
    #training_df['td_team'] = 'TD' + training_df['td_team'].astype(str)
    training_df['passer_player_id'] = 'Pass2' + training_df['passer_player_id'].astype(str)
    training_df['receiver_player_id'] = 'Rec' + training_df['receiver_player_id'].astype(str)
    training_df['rusher_player_id'] = 'Rush' + training_df['rusher_player_id'].astype(str)
    training_df['kicker_player_id'] = 'Kick' + training_df['kicker_player_id'].astype(str)
    #training_df['penalty_team'] = 'PTeam' + training_df['penalty_team'].astype(str)
    training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
    training_df['WDirection'] = training_df['WDirection'].astype(str)
    training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
    training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
    training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
    training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
    training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    #training_df['fumbled_1_player_id'] = 'fum' + training_df['fumbled_1_player_id'].astype(str)

    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    Aoff = list(pd.get_dummies(training_df['AOffense']).columns.values)
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['penalty_team'])], axis=1)
    #training_df = training_df.drop(columns=['penalty_team'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['kicker_player_id'])], axis=1)
    training_df = training_df.drop(columns=['kicker_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['rusher_player_id'])], axis=1)
    training_df = training_df.drop(columns=['rusher_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['receiver_player_id'])], axis=1)
    training_df = training_df.drop(columns=['receiver_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['passer_player_id'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['td_team'])], axis=1)
    #training_df = training_df.drop(columns=['td_team'])
    timeOuts = list(pd.get_dummies(training_df['timeout_team']).columns.values)#############################################3
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)
    training_df = training_df.drop(columns=['timeout_team'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['two_point_conv_result'])], axis=1)
    training_df = training_df.drop(columns=['two_point_conv_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['extra_point_result'])], axis=1)
    training_df = training_df.drop(columns=['extra_point_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['field_goal_result'])], axis=1)
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_gap'])], axis=1)
    training_df = training_df.drop(columns=['run_gap'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_location'])], axis=1)
    training_df = training_df.drop(columns=['pass_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam_type'])], axis=1)
    training_df = training_df.drop(columns=['posteam_type'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['fumbled_1_player_id'])], axis=1)
    training_df = training_df.drop(columns=['fumbled_1_player_id'])
    #training_df = training_df.drop(columns=['goal_to_go'])
    return training_df

def airYardsthrowOut(train_stats):
    for c in train_stats.columns.values.astype(str):
        #print(c)
        #print(type(c))
        #if c == '0':
        #    train_stats = train_stats.drop(c, axis=1)
        if c == 'yards_gained':
            train_stats = train_stats.drop(c, axis=1)
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Pass'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'air_yards':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Run'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Gap'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Field'):
        #   train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Extra'):
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('Two'):
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        #elif c.startswith('TO'):
        #    train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('TD'):
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'field_goal_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'punt_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Pass2'):
           train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Rec'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Rush'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Kick'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('fum'):
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        return train_stats


def trainAirYards(df):
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    df['passLocation'] = 0
    df.loc[df.pass_location == 'left', 'passLocation'] = 1
    df.loc[df.pass_location == 'middle', 'passLocation'] = 2
    df.loc[df.pass_location == 'right', 'passLocation'] = 3
    df = df.drop(columns=['pass_location'])

    #Change String stats to dummy columns
    df = clean(df)
    #Throw Out stats that are 'illegal'
    df = throwOut(df)

    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('air_yards', axis=1)
    y = df['air_yards']

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    from sklearn.ensemble import RandomForestClassifier

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 


def incompleteclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def incompletethrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'pass_location':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'incomplete_pass':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_gap':
            train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainIncompletePass(df):
    #df = pd.read_csv('runLocation.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    df['passLocation'] = 0
    df.loc[df.pass_location == 'left', 'passLocation'] = 1
    df.loc[df.pass_location == 'middle', 'passLocation'] = 2
    df.loc[df.pass_location == 'right', 'passLocation'] = 3
    df = df.drop(columns=['pass_location'])

    #Change String stats to dummy columns
    df = incompleteclean(df)
    #Throw Out stats that are 'illegal'
    df = incompletethrowOut(df)

    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('incomplete_pass', axis=1)
    y = df['incomplete_pass']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)

def passLocationclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def passLocationthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'pass_location':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'qb_hit':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'sack':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_gap':
            train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats

def trainPassLocation(df):
    #df = pd.read_csv('runLocation.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    df['passLocation'] = 0
    df.loc[df.pass_location == 'left', 'passLocation'] = 1
    df.loc[df.pass_location == 'middle', 'passLocation'] = 2
    df.loc[df.pass_location == 'right', 'passLocation'] = 3
    df = df.drop(columns=['pass_location'])

    #Change String stats to dummy columns
    df = passLocationclean(df)
    #Throw Out stats that are 'illegal'
    df = passLocationthrowOut(df)

    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('passLocation', axis=1)
    y = df['passLocation']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)

def sackclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
    training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
    training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
    training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])
    
    training_df = training_df.drop(columns=['run_gap'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_location'])], axis=1)
    training_df = training_df.drop(columns=['pass_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    timeOuts = list(pd.get_dummies(training_df['timeout_team']).columns.values)#############################################3
    training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)

    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    return training_df


def sackthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        #elif c == 'air_yards':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_location':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'kick_distance':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'timeout':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'qb_hit':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'sack':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_gap':
        #    train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainSack(df):
    #df = pd.read_csv('sack.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = sackclean(df)
    #Throw Out stats that are 'illegal'
    df = sackthrowOut(df)

    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('sack', axis=1)
    y = df['sack']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)


def qbHitclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
    training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
    training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
    training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])
    
    training_df = training_df.drop(columns=['run_gap'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_location'])], axis=1)
    training_df = training_df.drop(columns=['pass_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    timeOuts = list(pd.get_dummies(training_df['timeout_team']).columns.values)#############################################3
    training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)

    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    return training_df


def qbHitthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_location':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'kick_distance':
        #    train_stats = train_stats.drop(c, axis=1)
        #elif c == 'timeout':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'qb_hit':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_gap':
        #    train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainQBHit(df):
    #df = pd.read_csv('qbHit.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = qbHitclean(df)
    #Throw Out stats that are 'illegal'
    df = qbHitthrowOut(df)

    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('qb_hit', axis=1)
    y = df['qb_hit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)
    
def runYardsclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def runYardsthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_location':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_gap':
        #    train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainRunYards(df):

    print(df.head())

    df['gap'] = 0
    df.loc[df.run_gap == 'end', 'gap'] = 1
    df.loc[df.run_gap == 'tackle', 'gap'] = 2
    df.loc[df.run_gap == 'gaurd', 'gap'] = 3
    df = df.drop(columns=['run_gap'])

    df['location'] = 0
    df.loc[df.run_location == 'left', 'location'] = 1
    df.loc[df.run_location == 'middle', 'location'] = 2
    df.loc[df.run_location == 'right', 'location'] = 3
    df = df.drop(columns=['run_location'])

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = runYardsclean(df)
    #Throw Out stats that are 'illegal'
    df = runYardsthrowOut(df)


    #Numeric Value for each play type
    #Pass       0
    #Runs       1
    #Field Goal 2
    #Punt       3
    df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('yards_gained', axis=1)
    y = df['yards_gained']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  

    regressor = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=0)  
    regressor.fit(X_train, y_train)  
    y_pred = regressor.predict(X_test)  

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
    return(regressor)

def runGapclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def runGapthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_gap':
        #    train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainRunGap(df):
    #df = pd.read_csv('playType.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = runGapclean(df)
    #Throw Out stats that are 'illegal'
    df = runGapthrowOut(df)


    #Numeric Value for each play type
    #Pass       0
    #Runs       1
    #Field Goal 2
    #Punt       3
    df['gap'] = 0
    df.loc[df.run_gap == 'end', 'gap'] = 1
    df.loc[df.run_gap == 'tackle', 'gap'] = 2
    df.loc[df.run_gap == 'gaurd', 'gap'] = 3
    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt', 'run_gap'])
    df = df.drop(columns=['run_gap'])

    X = df.drop('gap', axis=1)
    y = df['gap']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)
    
def runLocationclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def runLocationthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        #elif c == 'run_location':
        #    train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_gap':
            train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainRunLocation(df):
    #df = pd.read_csv('runLocation.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = runLocationclean(df)
    #Throw Out stats that are 'illegal'
    df = runLocationthrowOut(df)

    #df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    df['location'] = 0
    df.loc[df.run_location == 'left', 'location'] = 1
    df.loc[df.run_location == 'middle', 'location'] = 2
    df.loc[df.run_location == 'right', 'location'] = 3
    df = df.drop(columns=['run_location'])

    X = df.drop('location', axis=1)
    y = df['location']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)


def mlclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def mlthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_gap':
            train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainml(df):
    #df = pd.read_csv('playType.csv')
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = mlclean(df)
    #Throw Out stats that are 'illegal'
    df = mlthrowOut(df)


    #Numeric Value for each play type
    #Pass       0
    #Runs       1
    #Field Goal 2
    #Punt       3
    df['play'] = 0
    df.loc[df.pass_attempt == 1, 'play'] = 1
    df.loc[df.rush_attempt == 1, 'play'] = 2
    df.loc[df.field_goal_attempt == 1, 'play'] = 3
    df.loc[df.punt_attempt == 1, 'play'] = 4
    df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('play', axis=1)
    y = df['play']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)

def spikeclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def spikethrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        #if c == 'qb_kneel':
        #    train_stats = train_stats.drop(c, axis=1)
        #if c == 'qb_spike':
        #   train_stats = train_stats.drop(c, axis=1)
        if c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_gap':
            train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainSpike(df):
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #df['try'] = 0
    #df = df.drop(columns=['extra_point_attempt', 'two_point_attempt'])


    #Change String stats to dummy columns
    df = spikeclean(df)
    #Throw Out stats that are 'illegal'
    df = spikethrowOut(df)

    df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('qb_spike', axis=1)
    y = df['qb_spike']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)

def kneeclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    

    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def kneethrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        #if c == 'qb_kneel':
        #    train_stats = train_stats.drop(c, axis=1)
        if c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_gap':
            train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout_team':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats


def trainKnee(df):
    #df = pd.read_csv('knee.csv')
    #df = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular')]

    #data = data.drop(df.index)

    #knee = data.loc[data['qb_kneel'] == 1]

    #data = data.sample(len(knee))

    #data2 = pd.concat([data, knee])

    #print(len(data))
    print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #df['try'] = 0
    df = df.drop(columns=['extra_point_attempt', 'two_point_attempt'])
    df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])


    #Change String stats to dummy columns
    df = kneeclean(df)
    #Throw Out stats that are 'illegal'
    df = kneethrowOut(df)

    X = df.drop('qb_kneel', axis=1)
    y = df['qb_kneel']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print(y_predict)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)

def timeOutclean(training_df):
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
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
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
    return training_df


def timeOutthrowOut(train_stats):
    '''
        This database contains stats that shouldn't be known by the model,
        so throw them out.
        (I.E. We shouldn't know who threw/caught the 
        ball before a pass play is even called, or if it's a scoring play,
        yards gained, etc.)
    '''
    for c in train_stats.columns.values.astype(str):
        if c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_location':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_rush':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'first_down_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'third_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_converted':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fourth_down_failed':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'incomplete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'interception':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'safety':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_lost':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'own_kickoff_recovery_td':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_hit':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'sack':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'pass_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rush_touchdown':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_attempt':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'complete_pass':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumble_recovery_1_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'return_yards':
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'duration':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'run_gap':
            train_stats = train_stats.drop(c, axis=1)   
        elif c == 'field_goal_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'two_point_conv_result':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'passer_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'rusher_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'receiver_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kicker_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'fumbled_1_player_id':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'extra_point_result':
            train_stats = train_stats.drop(c, axis=1)

        #elif c == 'timeout_team':
        #    train_stats = train_stats.drop(c, axis=1)
    return train_stats
    
def trainTimeOut(df):
    #df = pd.read_csv('timeout.csv')
    #df = df.loc[df['posteam'] != '0']
    #df = df.loc[(df['Year'] < 2018) | (df['SType'] == 'Pre')]

    #print(len(df))

    #to = df.loc[(df['timeout_team'] != '0')]

    #print(len(to))

    #df = df.drop(to.index)

    #df = df.sample(len(to))

    #data = pd.concat([df, to])

    #print(len(df))
    #print(df.head())

    df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    df = timeOutclean(df)
    #Throw Out stats that are 'illegal'
    df = timeOutthrowOut(df)

    df = df.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])

    X = df.drop('timeout_team', axis=1)
    y = df['timeout_team']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

    random_forest.fit(X_train, y_train)

    y_predict = random_forest.predict(X_test)
    print(y_predict)
    print("Accuracy")
    ab = accuracy_score(y_test, y_predict)
     
    print(ab) 
    return(random_forest)



def clean(training_df):
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
    training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
    training_df['extra_point_result'] = 'Extra' + training_df['extra_point_result'].astype(str)
    training_df['two_point_conv_result'] = 'Two' + training_df['two_point_conv_result'].astype(str)
    training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
    training_df['field_goal_result'] = 'Field' + training_df['field_goal_result'].astype(str)
    training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
    #training_df['td_team'] = 'TD' + training_df['td_team'].astype(str)
    training_df['passer_player_id'] = 'Pass2' + training_df['passer_player_id'].astype(str)
    training_df['receiver_player_id'] = 'Rec' + training_df['receiver_player_id'].astype(str)
    training_df['rusher_player_id'] = 'Rush' + training_df['rusher_player_id'].astype(str)
    training_df['kicker_player_id'] = 'Kick' + training_df['kicker_player_id'].astype(str)
    #training_df['penalty_team'] = 'PTeam' + training_df['penalty_team'].astype(str)
    training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
    training_df['WDirection'] = training_df['WDirection'].astype(str)
    training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
    training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
    training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
    training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
    training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='home', 'posteam_type']=1
    training_df['posteam_type'] = training_df.loc[training_df.posteam_type=='away', 'posteam_type']=0
    #training_df['fumbled_1_player_id'] = 'fum' + training_df['fumbled_1_player_id'].astype(str)

    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    #training_df[''] = '' + training_df[''].astype(str)
    Aoff = list(pd.get_dummies(training_df['AOffense']).columns.values)
    
    training_df = pd.concat([training_df, pd.get_dummies(training_df['SType'])], axis=1)
    training_df = training_df.drop(columns=['SType'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['AOffense'])], axis=1)
    training_df = training_df.drop(columns=['AOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ADefense'])], axis=1)
    training_df = training_df.drop(columns=['ADefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['ACoach'])], axis=1)
    training_df = training_df.drop(columns=['ACoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HOffense'])], axis=1)
    training_df = training_df.drop(columns=['HOffense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HDefense'])], axis=1)
    training_df = training_df.drop(columns=['HDefense'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['HCoach'])], axis=1)
    training_df = training_df.drop(columns=['HCoach'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['WDirection'])], axis=1)
    training_df = training_df.drop(columns=['WDirection'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['Weather'])], axis=1)
    training_df = training_df.drop(columns=['Weather'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['penalty_team'])], axis=1)
    #training_df = training_df.drop(columns=['penalty_team'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['kicker_player_id'])], axis=1)
    training_df = training_df.drop(columns=['kicker_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['rusher_player_id'])], axis=1)
    training_df = training_df.drop(columns=['rusher_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['receiver_player_id'])], axis=1)
    training_df = training_df.drop(columns=['receiver_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['passer_player_id'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_id'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['td_team'])], axis=1)
    #training_df = training_df.drop(columns=['td_team'])
    timeOuts = list(pd.get_dummies(training_df['timeout_team']).columns.values)#############################################3
    training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)
    #training_df = training_df.drop(columns=['timeout_team'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['two_point_conv_result'])], axis=1)
    training_df = training_df.drop(columns=['two_point_conv_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['extra_point_result'])], axis=1)
    training_df = training_df.drop(columns=['extra_point_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['field_goal_result'])], axis=1)
    training_df = training_df.drop(columns=['field_goal_result'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_gap'])], axis=1)
    training_df = training_df.drop(columns=['run_gap'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_location'])], axis=1)
    training_df = training_df.drop(columns=['pass_location'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
    training_df = training_df.drop(columns=['defteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
    training_df = training_df.drop(columns=['posteam'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam_type'])], axis=1)
    training_df = training_df.drop(columns=['posteam_type'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['fumbled_1_player_id'])], axis=1)
    training_df = training_df.drop(columns=['fumbled_1_player_id'])
    #training_df = training_df.drop(columns=['goal_to_go'])
    return training_df

if(1==1):
    
    #spikeM = trainSpike()
    #mlM = trainml()
    #runLocationM = trainRunLocation()
    #runGapM = trainRunGap()
    #runYardsM = trainRunYards(data)
    #QBHitM = trainQBHit()
    #sackM = trainSack()
    #passLocationM = trainPassLocation()
    #incompleteM = trainIncompletePass()
    #airYardsM = trainAirYards()
    #interceptionM = trainInterception()
    #passYardsM = trainPassYards()
    #fumbleM = trainFumble()
    #fumbleLostM = trainFumbleLost()
    #durationM = trainDuration()
    #trainPos() 
    #fieldGoalM = trainFieldGoal()
    ##############################################
    #   Simulation
    ##############################################

    data = pd.read_csv("testLast2.csv", low_memory=False)
    dirt = pd.read_csv("testLast2.csv", low_memory=False)
    data.loc[data.WTemp == '39/53', 'WTemp'] = '46'
    dirt.loc[dirt.WTemp == '39/53', 'WTemp'] = '46'
    data.drop(columns=['Index'])
    dirt.drop(columns=['Index'])
    timeOutM = trainTimeOut(data)
    kneeM = trainKnee(data)
    spikeM = trainSpike(data)
    mlM = trainml(data)
    runLocationM = trainRunLocation(data)
    runGapM = trainRunGap(data)
    runYardsM = trainRunYards(data)
    QBHitM = trainQBHit(data)
    sackM = trainSack(data)
    passLocationM = trainPassLocation(data)
    incompleteM = trainIncompletePass(data)
    passYardsM = trainPassYards(data)
    posM = trainPos(17, data) 
    
    #data.to_csv('why.csv')
    print("GG")
    #year = data['Year'].iloc[0]
    #game_id = data['game_id'].iloc[0]
    #posteam = data['posteam'].iloc[0]
    #defteam = data['defteam'].iloc[0]
    print("GG")
    knee = data.drop(columns=['extra_point_attempt', 'two_point_attempt'])##############################################
    knee = knee.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])
    knee = kneeclean(knee)
    knee = kneethrowOut(knee)
    print("PRE")
    knee = knee.drop(columns=['qb_kneel'])
    knee = knee.loc[(knee['Year'] == 2018) & (knee['Regular'] == 1) & (knee['game_seconds_remaining'] == 3600)]
    print(knee)
    print("^KNEE")
    tst = knee.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = kneeM.predict(tst)#####################
    timeOut = data.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])###############################
    timeOut = timeOutclean(timeOut)
    timeOut = timeOutthrowOut(timeOut)
    print("PRE")
    timeOut = timeOut.drop(columns=['timeout_team'])
    timeOut = timeOut.loc[(timeOut['Year'] == 2018) & (timeOut['Regular'] == 1) & (timeOut['game_seconds_remaining'] == 3600)]
    print(timeOut)
    print("^TO")
    tst = timeOut.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = timeOutM.predict(tst)
    print(a)
    print(a[0])
    print("POST")#################################################
    spike = data.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])#################################
    spike = spikeclean(spike)
    spike = spikethrowOut(spike)
    print("PRE")
    spike = spike.drop(columns=['qb_spike'])
    spike = spike.loc[(spike['Year'] == 2018) & (spike['Regular'] == 1) & (spike['game_seconds_remaining'] == 3600)]
    print(spike)
    print("^TO")
    tst = spike.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = spikeM.predict(tst)
    print(a)
    print(a[0])
    print("POST")#############################
    ml = data.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])#############################
    ml['play'] = 0
    ml.loc[ml.pass_attempt == 1, 'play'] = 1
    ml.loc[ml.rush_attempt == 1, 'play'] = 2
    ml.loc[ml.field_goal_attempt == 1, 'play'] = 3
    ml.loc[ml.punt_attempt == 1, 'play'] = 4
    ml = mlclean(ml)
    ml = mlthrowOut(ml)
    print("PRE")
    ml = ml.drop(columns=['play'])
    ml = ml.loc[(ml['Year'] == 2018) & (ml['Regular'] == 1) & (ml['game_seconds_remaining'] == 3600)]
    print(ml)
    print("^TO")
    tst = ml.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = mlM.predict(tst)
    print(a)
    print(a[0])
    print("POST")#############################
    runLocation = data.drop(columns=['run_location'])########################################
    runLocation = runLocationclean(runLocation)
    runLocation = runLocationthrowOut(runLocation)
    print("PRE")
    runLocation = runLocation.loc[(runLocation['Year'] == 2018) & (runLocation['Regular'] == 1) & (runLocation['game_seconds_remaining'] == 3600)]
    print(runLocation)
    print("^TO")
    tst = runLocation.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = runLocationM.predict(tst)
    print(a)
    print(a[0])##########################################################
    print("POST")############################################################
    runGap = data.drop(columns=['run_gap'])
    runGap = runGapclean(runGap)
    runGap = runGapthrowOut(runGap)
    print("PRE")
    runGap = runGap.loc[(runGap['Year'] == 2018) & (runGap['Regular'] == 1) & (runGap['game_seconds_remaining'] == 3600)]
    print(runGap)
    print("^TO")
    tst = runGap.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = runGapM.predict(tst)
    print(a)
    print(a[0])
    print("POST")################################################
    runYards = data.copy()#########################################
    runYards['gap'] = 0
    runYards.loc[runYards.run_gap == 'end', 'gap'] = 1
    runYards.loc[runYards.run_gap == 'tackle', 'gap'] = 2
    runYards.loc[runYards.run_gap == 'gaurd', 'gap'] = 3
    runYards = runYards.drop(columns=['run_gap'])

    runYards['location'] = 0
    runYards.loc[runYards.run_location == 'left', 'location'] = 1
    runYards.loc[runYards.run_location == 'middle', 'location'] = 2
    runYards.loc[runYards.run_location == 'right', 'location'] = 3
    runYards = runYards.drop(columns=['run_location'])

    runYards.loc[runYards.WTemp == '39/53', 'WTemp'] = '46'

    #Change String stats to dummy columns
    runYards = runYardsclean(runYards)
    #Throw Out stats that are 'illegal'
    runYards = runYardsthrowOut(runYards)


    #Numeric Value for each play type
    #Pass       0
    #Runs       1
    #Field Goal 2
    #Punt       3
    runYards = runYards.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt', 'yards_gained'])
    print("PRE")
    runYards = runYards.loc[(runYards['Year'] == 2018) & (runYards['Regular'] == 1) & (runYards['game_seconds_remaining'] == 3600)]
    print(runYards)
    print("^TO")
    tst = runYards.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = runYardsM.predict(tst)
    print(a)
    print(a[0])############################################################3
    print("POST")############################################################
    QBHit = data.copy()
    QBHit = qbHitclean(QBHit)
    QBHit = qbHitthrowOut(QBHit)
    QBHit = QBHit.drop(columns=['qb_hit'])
    print("PRE")
    QBHit = QBHit.loc[(QBHit['Year'] == 2018) & (QBHit['Regular'] == 1) & (QBHit['game_seconds_remaining'] == 3600)]
    print(QBHit)
    print("^TO")
    tst = QBHit.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = QBHitM.predict(tst)
    print(a)
    print(a[0])
    print("POST")################################################
    print("POST")############################################################
    sack = data.copy()
    sack = sackclean(sack)
    sack = sackthrowOut(sack)
    sack = sack.drop(columns=['qb_hit'])
    print("PRE")
    sack = sack.loc[(sack['Year'] == 2018) & (sack['Regular'] == 1) & (sack['game_seconds_remaining'] == 3600)]
    print(sack)
    print("^TO")
    tst = sack.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = sackM.predict(tst)
    print(a)
    print(a[0])
    print("POST")################################################
    print("POST")############################################################
    passLocation = data.copy()
    passLocation['passLocation'] = 0
    passLocation.loc[passLocation.pass_location == 'left', 'passLocation'] = 1
    passLocation.loc[passLocation.pass_location == 'middle', 'passLocation'] = 2
    passLocation.loc[passLocation.pass_location == 'right', 'passLocation'] = 3
    passLocation = passLocation.drop(columns=['pass_location'])
    passLocation = passLocationclean(passLocation)
    passLocation = passLocationthrowOut(passLocation)
    passLocation = passLocation.drop(columns=['passLocation'])
    print("PRE")
    passLocation = passLocation.loc[(passLocation['Year'] == 2018) & (passLocation['Regular'] == 1) & (passLocation['game_seconds_remaining'] == 3600)]
    print(passLocation)
    print("^TO")
    tst = passLocation.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = passLocationM.predict(tst)
    print(a)
    print(a[0])
    print("POST")################################################
    print("POST")############################################################
    incomplete = data.copy()
    incomplete['passLocation'] = 0
    incomplete.loc[incomplete.pass_location == 'left', 'incomplete'] = 1
    incomplete.loc[incomplete.pass_location == 'middle', 'incomplete'] = 2
    incomplete.loc[incomplete.pass_location == 'right', 'incomplete'] = 3
    incomplete = incomplete.drop(columns=['pass_location'])
    incomplete = incompleteclean(incomplete)
    incomplete = incompletethrowOut(incomplete)
    incomplete = incomplete.drop(columns=['incomplete_pass', 'passLocation'])
    print("PRE")
    incomplete = incomplete.loc[(incomplete['Year'] == 2018) & (incomplete['Regular'] == 1) & (incomplete['game_seconds_remaining'] == 3600)]
    print(incomplete)
    print("^TO")
    tst = incomplete.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = incompleteM.predict(tst)
    print(a)
    print(a[0])
    print("POST")################################################
    print("POST")############################################################
    passYards = data.copy()

    passYards = passYardsclean(passYards)
    passYards = passYardsthrowOut(passYards)
    passYards = passYards.drop(columns=['yards_gained'])
    print("PRE")
    passYards = passYards.loc[(passYards['Year'] == 2018) & (passYards['Regular'] == 1) & (passYards['game_seconds_remaining'] == 3600)]
    print(passYards)
    print("^TO")
    tst = passYards.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = passYardsM.predict(tst)
    print(a)
    print(a[0])
    print("POST")################################################
    print("POST")############################################################
    passYards = data.copy()

    passYards = passYardsclean(passYards)
    passYards = passYardsthrowOut(passYards)
    passYards = passYards.drop(columns=['yards_gained'])
    print("PRE")
    passYards = passYards.loc[(passYards['Year'] == 2018) & (passYards['Regular'] == 1) & (passYards['game_seconds_remaining'] == 3600)]
    print(passYards)
    print("^TO")
    tst = passYards.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = passYardsM.predict(tst)
    print(a)
    print(a[0])
    print("POST")################################################
    print("POST")############################################################
    passYards = data.copy()

    passYards = passYardsclean(passYards)
    passYards = passYardsthrowOut(passYards)
    passYards = passYards.drop(columns=['yards_gained'])
    print("PRE")
    passYards = passYards.loc[(passYards['Year'] == 2018) & (passYards['Regular'] == 1) & (passYards['game_seconds_remaining'] == 3600)]
    print(passYards)
    print("^TO")
    tst = passYards.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = passYardsM.predict(tst)
    print(a)
    print(a[0])
    print("POST")################################################
    print("POST")############################################################
    pos = data.copy()

    pos['passLocation'] = 0
    pos.loc[pos.pass_location == 'left', 'passLocation'] = 1
    pos.loc[pos.pass_location == 'middle', 'passLocation'] = 2
    pos.loc[pos.pass_location == 'right', 'passLocation'] = 3
    pos = pos.drop(columns=['pass_location'])

    pos['gap'] = 0
    pos.loc[pos.run_gap == 'end', 'gap'] = 1
    pos.loc[pos.run_gap == 'tackle', 'gap'] = 2
    pos.loc[pos.run_gap == 'gaurd', 'gap'] = 3
    pos = pos.drop(columns=['run_gap'])

    pos['location'] = 0
    pos.loc[pos.run_location == 'left', 'location'] = 1
    pos.loc[pos.run_location == 'middle', 'location'] = 2
    pos.loc[pos.run_location == 'right', 'location'] = 3
    pos = pos.drop(columns=['run_location'])    

    pos = posclean(pos)
    pos = posthrowOut(pos)
    pos = pos.drop(columns=['yards_gained'])
    print("PRE")
    pos = pos.loc[(pos['Regular'] == 1) & (pos['game_seconds_remaining'] == 3600)]
    print(pos)
    print("^TO")
    tst = pos.iloc[0]
    tst = tst.values.reshape(1, -1)
    print(tst)
    a = passYardsM.predict(tst)
    print(a)
    print(a[0])
    print("POST")################################################
    
    #data = data.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])
    
    
    pos = pos.loc[(pos['Regular'] == 1) & (pos['game_seconds_remaining'] == 3600)]
    data = data.loc[(data['Regular'] == 1) & (data['game_seconds_remaining'] == 3600)]
    dirt = data.loc[(data['Regular'] == 1) & (data['game_seconds_remaining'] == 3600)]
    
    
    games = data.copy()
    #games = clean(games)
    
    #Every Game in 2018
    for index, row in data.iterrows():
        
        time = 3600
        row['yardline_100'] = 75
        gameLog = row
        gameLog['ydstogo'] = 10
        
            

            
            
        if gameLog['posteam_type'] == 0:
            home = defteam
            away = posteam
        else:
            home = posteam
            away = defteam
        homeHit = 0
        awayHit = 0
        home1stRush = 0
        away1stRush = 0
        home1stPass = 0
        away1stPass = 0
        homeIncomplete = 0
        awayIncomplete = 0
        homeComplete = 0
        awayComplete = 0
        homeInt = 0
        awayInt = 0
        home3rdDownConv = 0
        away3rdDownConv = 0
        home4thDownConv = 0
        away4thDownConv = 0
        home3rdDownFail = 0
        away3rdDownFail = 0
        home4thDownFail = 0
        away4thDownFail = 0
        homePassTD = 0
        awayPassTD = 0
        homeRushTD = 0
        awayRushTD = 0
        homeFumble = 0
        awayFumble = 0
            
        #SIM GAME
        while(time > 0):
            print(gameLog)
            gameLog['timeout_team'] = 0#timeOutM.predict(gameLog.drop('timeout_team'))
            print(gameLog['timeout_team'])####################FILL IN
            if(gameLog['timeout_team'] != 0):
                if(gameLog['timeout_team'] == gameLog['posteam']):
                    gameLog['home_timeouts_remaining'] -= 1
                else:
                    gameLog['away_timeouts_remaining'] -= 1
            
            gameLog['qb_kneel'] = kneeM.predict(data.drop('qb_kneel', axis=1))########################FILL IN
            print(gameLog['qb_kneel'])
            if(gameLog['qb_kneel'] == 1):
                gameLog['duration'] = -40
                gameLog['yards_gained'] = 0
            else:
                gameLog['qb_spike'] = 0###################FILL IN
                if(gameLog['qb_spike'] == 1):
                    gameLog['yards_gained'] = 0
                    gameLog['duration'] = -5
                else:
                    gameLog['play'] = 1####################FILL IN
                    if(gameLog['play'] == 1):#Pass
                        gameLog['pass_attempt'] = 1
                        gameLog['rush_attempt'] = 0
                        gameLog['field_goal_attempt'] = 0
                        gameLog['punt_attempt'] = 0
                        gameLog['run_location'] = '0'
                        gameLog['run_gap'] = '0'
                        gameLog['kick_distance'] = 0
                        
                        gameLog['qb_hit'] = 1#################FILL IN
                        if gameLog['qb_hit'] == 1:
                            if(gameLog['posteam_type'] == 0):
                                awayHit += 1
                                gameLog['totHit'] = awayHit
                            else:
                                homeHit += 1
                                gameLog['totHit'] = homeHit
                        gameLog['sack'] = 0##################FILL IN
                        if(gameLog['sack'] == 0):
                            gameLog['passLocation'] = 1#################FILL IN
                            if(gameLog['passLocation'] == 1):
                                gameLog['pass_location'] = 'left'
                            elif(gameLog['passLocation'] == 2):
                                gameLog['pass_location'] = 'middle'
                            elif(gameLog['passLocation'] == 3):
                                gameLog['pass_location'] = 'right'
                            gameLog['incomplete_pass'] = 0###################FILL IN
                            if(gameLog['incomplete_pass'] == 1):
                                gameLog['complete_pass'] = 0
                                if(gameLog['posteam_type'] == 0):
                                    awayIncomplete += 1
                                    gameLog['totincomplete_pass'] = awayIncomplete
                                else:
                                    homeIncomplete += 1
                                    gameLog['totincomplete_pass'] = homeIncomplete
                            else:
                                if(gameLog['posteam_type'] == 0):
                                    awayComplete += 1
                                    gameLog['totcomplete_pass'] = awayComplete
                                else:
                                    homeComplete += 1
                                    gameLog['totcomplete_pass'] = homeComplete
                                gameLog['complete_pass'] = 1
                                #gameLog['air_yards'] = 5######################FILL IN
                                #gameLog['interception'] = 0#######################FILL IN
                                #if(gameLog['interception'] == 1):
                                #    if(gameLog['posteam_type'] == 0):
                                #        awayInt += 1
                                #        gameLog['totinterception'] = awayInt
                                #    else:
                                #        homeInt += 1
                                #        gameLog['totinterception'] = homeInt
                                #    save = gameLog['posteam']
                                #    gameLog['posteam'] = gameLog['defteam']
                                #    gameLog['defteam'] = save
                                #    save = gameLog['posteam_score']
                                #    gameLog['posteam_score'] = gameLog['defteam_score']
                                #    gameLog['defteam_score'] = save
                                #    
                                #    gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                                #    if(gameLog['posteam_type'] == 0):
                                #        gameLog['totinterception'] = awayInt
                                #        gameLog['totHit'] = awayHit
                                #        gameLog['totfirst_down_rush'] = away1stRush
                                #        gameLog['totfirst_down_pass'] = away1stPass
                                #        gameLog['totincomplete_pass'] = awayIncomplete
                                #        gameLog['totcomplete_pass'] = awayComplete
                                #        gameLog['totinterception'] = awayInt
                                #        gameLog['totThirdDown_convert'] = away3rdDownConv
                                #        gameLog['totThirdDown_fail'] = away3rdDownFail
                                #        gameLog['totFourthDown_fail'] = away4thDownFail
                                #        gameLog['fourth_down_converted'] = away4thDownConv
                                #        gameLog['totPass_TD'] = awayPassTD
                                #        gameLog['totRush_TD'] = awayRushTD
                                #        gameLog['totfumble'] = awayFumble
                                #    else:
                                #        gameLog['totinterception'] = homeInt
                                #        gameLog['totHit'] = homeHit
                                #        gameLog['totfirst_down_rush'] = home1stRush
                                #        gameLog['totfirst_down_pass'] = home1stPass
                                #        gameLog['totincomplete_pass'] = homeIncomplete
                                #        gameLog['totcomplete_pass'] = homeComplete
                                #        gameLog['totinterception'] = homeInt
                                #        gameLog['totThirdDown_convert'] = home3rdDownConv
                                #        gameLog['totThirdDown_fail'] = home3rdDownFail
                                #        gameLog['totFourthDown_fail'] = home4thDownFail
                                #        gameLog['fourth_down_converted'] = home4thDownConv
                                #        gameLog['totPass_TD'] = homePassTD
                                #        gameLog['totRush_TD'] = homeRushTD
                                #        gameLog['totfumble'] = homeFumble
                                #        
                                #    gameLog['down'] = 1
                                #    gameLog['yardline_100'] = 100 - gameLog['yardline_100']
                                #    if(gameLog['yardline_100'] < 10):
                                #        gameLog['ydstogo'] = gameLog['yardline_100']
                                #    else:
                                #        gameLog['ydstogo'] = 10
                                gameLog['yards_gained'] = 7#######################FILL IN  
                                gameLog['yardline_100'] = gameLog['yardline_100'] - gameLog['yards_gained']

                                gameLog['ydstogo'] = gameLog['ydstogo'] - gameLog['yards_gained']
                                if(gameLog['yardline_100'] <= 0):
                                    if(gameLog['posteam_type'] == 0):
                                        awayPassTD += 1
                                        gameLog['totPass_TD'] = awayPassTD
                                        gameLog['total_away_score'] += 7
                                    else:
                                        homePassTD += 1
                                        gameLog['totPass_TD'] = homePassTD
                                        gameLog['total_home_score'] +=7
                                    gameLog['posteam_score'] += 7
                                    
                                    save = gameLog['posteam']#####################
                                    gameLog['posteam'] = gameLog['defteam']
                                    gameLog['defteam'] = save
                                    save = gameLog['posteam_score']
                                    gameLog['posteam_score'] = gameLog['defteam_score']
                                    gameLog['defteam_score'] = save
                                    gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                                    gameLog['down'] = 1
                                    gameLog['yardline_100'] = 75
                                    if(gameLog['yardline_100'] < 10):
                                        gameLog['ydstogo'] = gameLog['yardline_100']
                                    else:
                                        gameLog['ydstogo'] = 10
                                    gameLog['score_differential'] = gameLog['posteam_score'] - gameLog['defteam']
                                        
                                        
                                        
                                        
                                        
                                if(gameLog['ydstogo'] <= 0):
                                    if(gameLog['posteam_type'] == 0):
                                        away1stPass += 1
                                        gameLog['totfirst_down_pass'] = away1stPass
                                        if(gameLog['down'] == 3):
                                            away3rdDownConv += 1
                                            gameLog['totThirdDown_convert'] = away3rdDownConv
                                        elif(gameLog['down'] == 4):
                                            away4thDownConv += 1
                                            gameLog['totFourthDown_convert'] = away4thDownConv
                                    else:
                                        home1stPass += 1
                                        gameLog['totfirst_down_pass'] = home1stPass
                                        if(gameLog['down'] == 3):
                                            home3rdDownConv += 1
                                            gameLog['totThirdDown_convert'] = home3rdDownConv
                                        elif(gameLog['down'] == 4):
                                            home4thDownConv += 1
                                            gameLog['totFourthDown_convert'] = home4thDownConv   
                                elif(gameLog['down'] == 3):
                                    if(gameLog['posteam_type'] == 0):
                                        away3rdDownFail += 1
                                        gameLog['totThirdDown_fail'] = away3rdDownFail
                                    else:
                                        home3rdDownFail += 1
                                        gameLog['totThirdDown_fail'] = home3rdDownFail
                                elif(gameLog['down'] == 4):
                                    if(gameLog['posteam_type'] == 0):
                                        away4thDownFail += 1
                                        gameLog['totThirdDown_fail'] = away4thDownFail
                                    else:
                                        home4thDownFail += 1
                                        gameLog['totThirdDown_fail'] = home4thDownFail
                                    save = gameLog['posteam']#####################
                                    gameLog['posteam'] = gameLog['defteam']
                                    gameLog['defteam'] = save
                                    save = gameLog['posteam_score']
                                    gameLog['posteam_score'] = gameLog['defteam_score']
                                    gameLog['defteam_score'] = save
                                    gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                                    gameLog['down'] = 1
                                    gameLog['yardline_100'] = 100 - gameLog['yardline_100']
                                    if(gameLog['yardline_100'] < 10):
                                        gameLog['ydstogo'] = gameLog['yardline_100']
                                    else:
                                        gameLog['ydstogo'] = 10
                                    gameLog['score_differential'] = gameLog['posteam_score'] - gameLog['defteam']
   
                                    
                                    

                                gameLog['kicker_player_id'] = 0
                                gameLog['rusher_player_id'] = 0
                                
                                gameLog['passer_player_id'] = 1###################FILL IN
                                gameLog['receiver_player_id'] = 1###############FILL IN
                                #gameLog['passer_player_id'] = PASStranslate(gameLog)#######################FILL IN
                                #gameLog['receiver_player_id'] = RECtranslate(gameLog)#######################FILL IN
                        
                    elif(gameLog['play'] == 2):#RUN DA BALL
                        gameLog['pass_attempt'] = 0
                        gameLog['rush_attempt'] = 1
                        gameLog['field_goal_attempt'] = 0
                        gameLog['punt_attempt'] = 0
                        gameLog['kick_distance'] = 0
                        gameLog['qb_hit'] = 0
                        gameLog['sack'] = 0
                        gameLog['passLocation'] = 0
                        gameLog['pass_location'] = '0'
                        gameLog['incomplete_pass'] = 0
                        gameLog['air_yards'] = 0
                        gameLog['interception'] = 0
                        
                        location = 1#########################FILL IN
                        if(location == 1):
                            gameLog['run_location'] = 'left'
                        elif(location == 2):
                            gameLog['run_location'] = 'middle'
                        elif(location == 3):
                            gameLog['run_location'] = 'right'
                        gap = 1
                        if(gap == 1):
                            gameLog['run_gap'] = 'end'
                        elif(gap == 2):
                            gameLog['run_gap'] = 'tackle'
                        elif(gap == 3):
                            gameLog['run_gap'] = 'guard'
                        gameLog['yards_gained'] = 5##########FILL IN
                        
                        gameLog['yardline_100'] = gameLog['yardline_100'] - gameLog['yards_gained']
                        
                        if(gameLog['yardline_100'] <= 0):
                            if(gameLog['posteam_type'] == 0):
                                awayRushTD += 1
                                gameLog['totRush_TD'] = awayRushTD
                                gameLog['total_away_score'] += 7
                            else:
                                homeRushTD += 1
                                gameLog['totRush_TD'] = homeRushTD
                                gameLog['total_home_score'] +=7
                            gameLog['posteam_score'] += 7
                                    
                            save = gameLog['posteam']#####################
                            gameLog['posteam'] = gameLog['defteam']
                            gameLog['defteam'] = save
                            save = gameLog['posteam_score']
                            gameLog['posteam_score'] = gameLog['defteam_score']
                            gameLog['defteam_score'] = save
                            gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                            gameLog['down'] = 1
                            gameLog['yardline_100'] = 75
                            if(gameLog['yardline_100'] < 10):
                                gameLog['ydstogo'] = gameLog['yardline_100']
                            else:
                                gameLog['ydstogo'] = 10
                            gameLog['score_differential'] = gameLog['posteam_score'] - gameLog['defteam']
                                        
                                        
                                        
                                        
                                        
                        if(gameLog['ydstogo'] <= 0):
                            if(gameLog['posteam_type'] == 0):
                                away1stPass += 1
                                gameLog['totfirst_down_rush'] = away1stRush
                                if(gameLog['down'] == 3):
                                    away3rdDownConv += 1
                                    gameLog['totThirdDown_convert'] = away3rdDownConv
                                elif(gameLog['down'] == 4):
                                    away4thDownConv += 1
                                    gameLog['totFourthDown_convert'] = away4thDownConv
                            else:
                                home1stPass += 1
                                gameLog['totfirst_down_rush'] = home1stRush
                                if(gameLog['down'] == 3):
                                    home3rdDownConv += 1
                                    gameLog['totThirdDown_convert'] = home3rdDownConv
                                elif(gameLog['down'] == 4):
                                    home4thDownConv += 1
                                    gameLog['totFourthDown_convert'] = home4thDownConv   
                        elif(gameLog['down'] == 3):
                            if(gameLog['posteam_type'] == 0):
                                away3rdDownFail += 1
                                gameLog['totThirdDown_fail'] = away3rdDownFail
                            else:
                                home3rdDownFail += 1
                                gameLog['totThirdDown_fail'] = home3rdDownFail
                        elif(gameLog['down'] == 4):
                            if(gameLog['posteam_type'] == 0):
                                away4thDownFail += 1
                                gameLog['totThirdDown_fail'] = away4thDownFail
                            else:
                                home4thDownFail += 1
                                gameLog['totThirdDown_fail'] = home4thDownFail
                            save = gameLog['posteam']#####################
                            gameLog['posteam'] = gameLog['defteam']
                            gameLog['defteam'] = save
                            save = gameLog['posteam_score']
                            gameLog['posteam_score'] = gameLog['defteam_score']
                            gameLog['defteam_score'] = save
                            gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                            gameLog['down'] = 1
                            gameLog['yardline_100'] = 100 - gameLog['yardline_100']
                            if(gameLog['yardline_100'] < 10):
                                gameLog['ydstogo'] = gameLog['yardline_100']
                            else:
                                gameLog['ydstogo'] = 10
                            gameLog['score_differential'] = gameLog['posteam_score'] - gameLog['defteam']
                        
                        
                        
                        gameLog['passer_player_id'] = 0
                        gameLog['receiver_player_id'] = 0
                        gameLog['kicker_player_id'] = 0
                        
                        gameLog['rusher_player_id'] = 1###############FILL IN
                        #gameLog['rusher_player_id'] = RUSHtranslate(gameLog)#######################FILL IN
                        
                            
                    elif(gameLog['play'] == 3):#FIELD GOAL
                        gameLog['pass_attempt'] = 0
                        gameLog['rush_attempt'] = 0
                        gameLog['field_goal_attempt'] = 1
                        gameLog['punt_attempt'] = 0
                        gameLog['qb_hit'] = 0
                        gameLog['sack'] = 0
                        gameLog['passLocation'] = 0
                        gameLog['pass_location'] = '0'
                        gameLog['incomplete_pass'] = 0
                        gameLog['air_yards'] = 0
                        gameLog['run_location'] = '0'
                        gameLog['run_gap'] = '0'  
                        gameLog['interception'] = 0
                        
                        gameLog['kick_distance'] = gameLog['yardline_100']+18
                        gameLog['passer_player_id'] = 0
                        gameLog['receiver_player_id'] = 0
                        gameLog['rusher_player_id'] = 0
                        
                        gameLog['kicker_player_id'] = 1#######################FILL IN
                        gameLog['field_goal_result'] = 1#####################FILL IN
                        #gameLog['kicker_player_id'] = Ktranslate(gameLog['kicker_player_id'])#######################FILL IN
                        if(gameLog['field_goal_result'] == 1):
                            if(gameLog['posteam_type'] == 0):
                                gameLog['total_home_score'] += 3
                            else:
                                gameLog['total_away_score'] += 3
                            gameLog['posteam_score'] += 3
                            save = gameLog['posteam']#####################
                            gameLog['posteam'] = gameLog['defteam']
                            gameLog['defteam'] = save
                            save = gameLog['posteam_score']
                            gameLog['posteam_score'] = gameLog['defteam_score']
                            gameLog['defteam_score'] = save
                            gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                            gameLog['down'] = 1
                            gameLog['yardline_100'] = 75
                            if(gameLog['yardline_100'] < 10):
                                gameLog['ydstogo'] = gameLog['yardline_100']
                            else:
                                gameLog['ydstogo'] = 10
                            gameLog['score_differential'] = gameLog['posteam_score'] - gameLog['defteam']
                        else:
                            save = gameLog['posteam']#####################
                            gameLog['posteam'] = gameLog['defteam']
                            gameLog['defteam'] = save
                            save = gameLog['posteam_score']
                            gameLog['posteam_score'] = gameLog['defteam_score']
                            gameLog['defteam_score'] = save
                            gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                            gameLog['down'] = 1
                            gameLog['yardline_100'] = 100 - gameLog['yardline_100']
                            if(gameLog['yardline_100'] < 10):
                                gameLog['ydstogo'] = gameLog['yardline_100']
                            else:
                                gameLog['ydstogo'] = 10
                            gameLog['score_differential'] = gameLog['posteam_score'] - gameLog['defteam']
                        
                        
                            
                        
                    elif(gameLog['play'] == 4):
                        gameLog['pass_attempt'] = 0
                        gameLog['rush_attempt'] = 0
                        gameLog['field_goal_attempt'] = 0
                        gameLog['punt_attempt'] = 1
                        gameLog['yards_gained'] = (100- gameLog['yardline_100'])/2
                        save = gameLog['posteam']#####################
                        gameLog['posteam'] = gameLog['defteam']
                        gameLog['defteam'] = save
                        save = gameLog['posteam_score']
                        gameLog['posteam_score'] = gameLog['defteam_score']
                        gameLog['defteam_score'] = save
                        gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                        gameLog['down'] = 1
                        gameLog['yardline_100'] = 100 - gameLog['yardline_100']
                        if(gameLog['yardline_100'] < 10):
                            gameLog['ydstogo'] = gameLog['yardline_100']
                        else:
                            gameLog['ydstogo'] = 10
                        gameLog['score_differential'] = gameLog['posteam_score'] - gameLog['defteam']            
                        
                        
      
                    gameLog['fumble'] = 0########################FILL IN
                    gameLog['fumble_lost'] = 0###################FILL IN
                    if(gameLog['fumble_lost'] == 1):
                        gameLog['fumble_recovery_1_yards'] = 3##################FILL IN
                        save = gameLog['posteam']
                        gameLog['posteam'] = gameLog['defteam']
                        gameLog['defteam'] = save
                        save = gameLog['posteam_score']
                        gameLog['posteam_score'] = gameLog['defteam_score']
                        gameLog['defteam_score'] = save
                        gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                        gameLog['down'] = 1
                        gameLog['yardline_100'] = 100 - gameLog['yardline_100']
                        if(gameLog['yardline_100'] < 10):
                            gameLog['ydstogo'] = gameLog['yardline_100']
                        else:
                            gameLog['ydstogo'] = 10
                    if(gameLog['yardline_100'] >= 100):#Safety
                        if(gameLog['posteam_type'] == 0):
                            gameLog['total_home_score'] += 2
                        else:
                            gameLog['total_away_score'] +=2
                            gameLog['defteam_score'] += 2
                            save = gameLog['posteam']#####################
                            gameLog['posteam'] = gameLog['defteam']
                            gameLog['defteam'] = save
                            save = gameLog['posteam_score']
                            gameLog['posteam_score'] = gameLog['defteam_score']
                            gameLog['defteam_score'] = save
                            gameLog['posteam_type'] = (gameLog['posteam_type']+1)%2
                            gameLog['down'] = 1
                            gameLog['yardline_100'] = 75
                            if(gameLog['yardline_100'] < 10):
                                gameLog['ydstogo'] = gameLog['yardline_100']
                            else:
                                gameLog['ydstogo'] = 10
                            gameLog['score_differential'] = gameLog['posteam_score'] - gameLog['defteam']
                    gameLog['duration'] = -40#FILL IN!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    
    
            gameLog['quarter_seconds_remaining'] = (gameLog['quarter_seconds_remaining'] + gameLog['duration'])
            gameLog['half_seconds_remaining'] = gameLog['half_seconds_remaining'] + gameLog['duration']
            gameLog['game_seconds_remaining'] = gameLog['game_seconds_remaining'] + gameLog['duration']
            time = time + gameLog['duration']
            #print(gameLog['quarter_seconds_remaining'])
            #print(gameLog['duration'])
            #print(time)
            if gameLog['quarter_seconds_remaining'] <= 0:
                if gameLog['qtr'] == 1:
                   gameLog['half_seconds_remaining'] = 900
                   gameLog['game_seconds_remaining'] = 2700
                   gameLog['qtr'] = 2
                   gameLog['quarter_seconds_remaining'] = 900
                   time = 2700
                elif gameLog['qtr'] == 2:
                    gameLog['half_seconds_remaining'] = 1800
                    gameLog['game_seconds_remaining'] = 1800
                    gameLog['qtr'] = 3
                    gameLog['quarter_seconds_remaining'] = 900
                    time = 1800
                elif gameLog['qtr'] == 3:
                    gameLog['half_seconds_remaining'] = 900
                    gameLog['game_seconds_remaining'] = 900
                    gameLog['qtr'] = 4
                    gameLog['quarter_seconds_remaining'] = 900
                    time = 900
                elif gameLog['qtr'] == 4:
                    if gameLog['score_differential'] == 0:
                        gameLog['half_seconds_remaining'] = 900
                        gameLog['game_seconds_remaining'] = 900
                        gameLog['qtr'] = 5
                        gameLog['quarter_seconds_remaining'] = 900
                        time = 900
                    else:  
                        gameLog['half_seconds_remaining'] = 0
                        gameLog['game_seconds_remaining'] = 0
                        gameLog['quarter_seconds_remaining'] = 0
                elif gameLog['qtr'] == 5:
                    gameLog['half_seconds_remaining'] = 0
                    gameLog['game_seconds_remaining'] = 0
                    gameLog['quarter_seconds_remaining'] = 0
                    
                        
                

            