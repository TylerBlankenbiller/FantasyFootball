import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path


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


def throwOut(train_stats):
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

teams = np.loadtxt('TeamNames.txt', dtype='str')
print(teams)
week = 1
while(week <= 17):
    for team in teams:
        data = pd.read_csv('testLast2.csv', low_memory=False)
        data = data.loc[(data['posteam'] != '0') & (data['posteam'] == team)]
        data = data.loc[(data['Year'] == 2018) & (data['SType'] == 'Regular') & (data['posteam'] == team) & (data['Week'] <= week)]
        #QB = data['passer_player_id'].unique()
        #RB = data['rusher_player_id'].unique()
        #WR = data['receiver_player_id'].unique()
        #K = data['kicker_player_id'].unique()
        
        df = pd.read_csv(team+'players.csv')
        
        df.loc[df.WTemp == '39/53', 'WTemp'] = '46'

        df = df.drop(columns=['passer_player_id_x', 'receiver_player_id_x', 'rusher_player_id_x', 'kicker_player_id_x',
                        'passer_player_id_y', 'receiver_player_id_y', 'rusher_player_id_y', 'kicker_player_id_y'])

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
        
        
        
        
        
        
        QBtrans = np.loadtxt(team+'QB.txt', dtype='str')
        RBtrans = np.loadtxt(team+'RB.txt', dtype='str')
        WRtrans = np.loadtxt(team+'WR.txt', dtype='str')
        my_file = Path('/'+team+'K.txt')
        if my_file.is_file():
            Ktrans = np.loadtxt(team+'K.txt', dtype='str')
        
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
                if(QBdf['THRpasser_player_id'][pindex] == QBdata['passer_player_id'][did]):
                    QBdf['THRpasser_player_id'][pindex] = QBdata['passer_player_id'][did]


        teams = QBdf['posteam'].unique()

        #Change String stats to dummy columns
        QBdf = clean(QBdf)
        #Throw Out stats that are 'illegal'
        QBdf = throwOut(QBdf)

        X = QBdf.drop('THRpasser_player_id', axis=1)
        y = QBdf['THRpasser_player_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

        random_forest.fit(X_train, y_train)

        y_predict = random_forest.predict(X_test)
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
                if(RBdf['RSHrusher_player_id'][pindex] == RBdata['rusher_player_id'][did]):
                    RBdf['RSHrusher_player_id'][pindex] = RBdata['rusher_player_id'][did]


        teams = RBdf['posteam'].unique()

        #Change String stats to dummy columns
        RBdf = clean(RBdf)
        #Throw Out stats that are 'illegal'
        RBdf = throwOut(RBdf)

        X = RBdf.drop('RSHrusher_player_id', axis=1)
        y = RBdf['RSHrusher_player_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

        random_forest.fit(X_train, y_train)

        y_predict = random_forest.predict(X_test)
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
                if(WRdf['RECreceiver_player_id'][pindex] == WRdata['receiver_player_id'][did]):
                    WRdf['RECreceiver_player_id'][pindex] = WRdata['receiver_player_id'][did]


        teams = WRdf['posteam'].unique()

        #Change String stats to dummy columns
        WRdf = clean(WRdf)
        #Throw Out stats that are 'illegal'
        WRdf = throwOut(WRdf)

        X = WRdf.drop('RECreceiver_player_id', axis=1)
        y = WRdf['RECreceiver_player_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

        random_forest.fit(X_train, y_train)

        y_predict = random_forest.predict(X_test)
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
                    if(Kdf['KICKkicker_player_id'][pindex] == Kdata['kicker_player_id'][did]):
                        Kdf['KICKkicker_player_id'][pindex] = Kdata['kicker_player_id'][did]


        teams = Kdf['posteam'].unique()

        #Change String stats to dummy columns
        Kdf = clean(Kdf)
        #Throw Out stats that are 'illegal'
        Kdf = throwOut(Kdf)

        X = Kdf.drop('KICKkicker_player_id', axis=1)
        y = Kdf['KICKkicker_player_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        random_forest = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=1)

        random_forest.fit(X_train, y_train)

        y_predict = random_forest.predict(X_test)
        print("Accuracy")
        ab = accuracy_score(y_test, y_predict)
         
        print(ab)  
    week+=1
    print('Week for players'0
    print(week)
    