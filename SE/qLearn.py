    
from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc

def clean(training_df):
    training_df = training_df.replace({'STL':'LA'}, regex=True)
    training_df = training_df.replace({'SD':'LAC'}, regex=True)
    training_df = training_df.replace({'JAC':'JAX'}, regex=True)
    training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
    training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
    training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
    #training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
    training_df['extra_point_result'] = 'Extra' + training_df['extra_point_result'].astype(str)
    training_df['two_point_conv_result'] = 'Two' + training_df['two_point_conv_result'].astype(str)
    #training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
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
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['run_gap'])], axis=1)
    #training_df = training_df.drop(columns=['run_gap'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    #training_df = training_df.drop(columns=['run_location'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['pass_location'])], axis=1)
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

def simSave(gameLog):
    gameLog['yards_gained'] = prediction[0]
    gameLog['qb_kneel'] = prediction[0]
    gameLog['qb_spike'] = prediction[0]
    gameLog['air_yards'] = prediction[0]
    gameLog['yards_after_catch'] = prediction[0]
    gameLog['field_goal_result'] = prediction[0]
    gameLog['kick_distance'] = prediction[0]
    gameLog['extra_point_result'] = prediction[0]
    gameLog['two_point_conv_result'] = prediction[0]
    gameLog['timeout'] = prediction[0]
    gameLog['timeout_team'] = prediction[0]
    gameLog['td_team'] = prediction[0]
    gameLog['first_down_rush'] = prediction[0]
    gameLog['first_down_pass'] = prediction[0]
    gameLog['third_down_converted'] = prediction[0]
    gameLog['third_down_failed'] = prediction[0]
    gameLog['fourth_down_converted'] = prediction[0]
    gameLog['fourth_down_failed'] = prediction[0]
    gameLog['incomplete_pass'] = prediction[0]
    gameLog['interception'] = prediction[0]
    gameLog['safety'] = prediction[0]
    gameLog['fumble_lost'] = prediction[0]
    gameLog['qb_hit'] = prediction[0]
    gameLog['rush_attempt'] = prediction[0]
    gameLog['pass_attempt'] = prediction[0]
    gameLog['sack'] = prediction[0]
    gameLog['touchdown'] = prediction[0]
    gameLog['pass_touchdown'] = prediction[0]
    gameLog['rush_touchdown'] = prediction[0]
    gameLog['extra_point_attempt'] = prediction[0]
    gameLog['two_point_attempt'] = prediction[0]
    gameLog['field_goal_attempt'] = prediction[0]
    gameLog['fumble'] = prediction[0]
    gameLog['complete_pass'] = prediction[0]
    gameLog['passer_player_id'] = prediction[0]
    gameLog['receiver_player_id'] = prediction[0]
    gameLog['rusher_player_id'] = prediction[0]
    gameLog['kicker_player_id'] = prediction[0]
    gameLog['fumbled_1_player_id'] = prediction[0]
    gameLog['fumble_recovery_1_yards'] = prediction[0]
    gameLog['return_yards'] = prediction[0]
    gameLog['duration'] = prediction[0]
    
    return gameLog

def throws(train_stats):
    for c, v in train_stats.items():
        #print(c)
        #print(type(c))
        #if c == '0':
        #    train_stats = train_stats.drop(c, axis=1)
        if c == 'yards_gained':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_kneel':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'qb_spike':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Pass'):
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Run'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Gap'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Field'):
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Extra'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Two'):
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('TO'):
            train_stats = train_stats.drop(c, axis=1)
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

def throwOut(train_stats):
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
        elif c.startswith('Pass'):
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'air_yards':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'yards_after_catch':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Run'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Gap'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Field'):
           train_stats = train_stats.drop(c, axis=1)
        elif c == 'kick_distance':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Extra'):
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('Two'):
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'timeout':
            train_stats = train_stats.drop(c, axis=1)
        elif c.startswith('TO'):
            train_stats = train_stats.drop(c, axis=1)
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

tf.enable_eager_execution()

#with tf.device("/device:GPU:0"):
if 1 == 1:
    #Test below, never tried
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config, ...)


    tf.enable_eager_execution()

    print(tf.__version__)
    #test.csv!!!!!!!!!!!!!!!!!!!!!!!!!!!@#@!#@!#@!#@!#!!@predict
    training_df: pd.DataFrame = pd.read_csv("rushYards.csv", low_memory=False)
    training_df.loc[training_df.WTemp == '39/53', 'WTemp'] = 46
    #training_df = training_df.loc[training_df['posteam'] != 0]
    #training_df = training_df.drop(columns=['game_id', 'THRpasser_player_id', 'RECreceiver_player_id', 'RSHrusher_player_id', 'KICKkicker_player_id'])
    training_df['location'] = 0
    training_df.loc[training_df.run_location == 'left', 'location'] = 1
    training_df.loc[training_df.run_location == 'middle', 'location'] = 2
    training_df.loc[training_df.run_location == 'right', 'location'] = 3
    training_df = training_df.drop(columns=['run_location'])
    
    training_df['gap'] = 0
    training_df.loc[training_df.run_gap == 'end', 'gap'] = 1
    training_df.loc[training_df.run_gap == 'tackle', 'gap'] = 2
    training_df.loc[training_df.run_gap == 'gaurd', 'gap'] = 3
    training_df = training_df.drop(columns=['run_gap'])
    
    training_df = clean(training_df)
    training_df = throwOut(training_df)
    training_df = training_df.loc[training_df.rush_attempt == 1]

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
    print("goodest")
    
    
    
    #train_stats = throwOut(train_stats)   
    train_stats = train_stats.pop('yards_gained')
    
    
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset
    test_labels = train_dataset
    train_labels = train_dataset.pop('yards_gained')
    test_labels = test_dataset.pop('yards_gained')

    print('Good')

    def build_model():
        model = keras.Sequential([
            layers.Dense(1012, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(512, activation='relu'),
            layers.Dense(1)])

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

    EPOCHS = 70

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
        plt.ylabel('Mean Abs Error [yards_gained]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label = 'Val Error')
        plt.ylim([-100,1000])
        plt.legend()
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$yards_gained^2$]')
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
    plt.xlabel('True Values [yards_gained]')
    plt.ylabel('Predictions [yards_gained]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([-100,plt.xlim()[1]])
    plt.ylim([-100,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [yards_gained]")
    _ = plt.ylabel("Count")
    
    
    ##############################################
    #   Simulation
    ##############################################
    del train_dataset
    del train_stats
    del train_labels
    del test_labels
    data = pd.read_csv("predict.csv", low_memory=False, index_col=0)

    games = data.copy()
    games = clean(games)
    games = games.drop(columns=['0'])
    games = throwOut(games)
    #games = clean(games)
    #games = throws(games)
    
    #Every Game in 2018
    for index, row in games.iterrows():
        year = data['Year'][index]
        game_id = data['game_id'][index]
        posteam = 'PHI'#data['posteam'][index]
        defteam = 'NYG'#data['defteam'][index]
        #row = throws(row)
        #print(row['c1'], row['c2'])
        time = 3600
        row['yardline_100'] = 75
        row = row.drop(labels=['game_id'])
        prediction = model.predict(row).flatten()
            
        for i in range(len(prediction)):
            print(i)
            print("i^")
            print(prediction[i])
            
        gameLog = temp#Fresh game
        gameLog = simSave(gameLog)#Add Predictions
            
        save = gameLog#Save this play
            
        if gameLog['posteam_type'] == 0:
            home = defteam
            away = posteam
        else:
            home = posteam
            away = defteam
        homeHit = 0
        awayHit = 0
        homeRush = 0
        awayRush = 0
        homePass = 0
        awayPass = 0
        homeIncomplete = 0
        awayIncomplete = 0
        homeInt = 0
        awayInt = 0
            
        while(time > 0):
            gameLog['quarter_seconds_remaining'] = gameLog['quarter_seconds_remaining'] + gameLog['duration']
            gameLog['half_seconds_remaining'] = gameLog['half_seconds_remaining'] + gameLog['duration']
            gameLog['game_seconds_remaining'] = gameLog['game_seconds_remaining'] + gameLog['duration']
            time = time + game['duration']
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
                    
                        
                        
            gameLog['yardline_100'] -= gameLog['yards_gained']
            if gameLog['posteam_type'] == 1:
                gameLog['totHit'] = homeHit = homeHit + gameLog['qb_hit']
                gameLog['totfirst_down_rush'] = homeRush = homeRush + gameLog['totfirst_down_rush']
                gameLog['totfirst_down_pass'] = homePass = homePass + gameLog['totfirst_down_pass']
                gameLog['totincomplete_pass'] = homeIncomplete = homeIncomplete + gameLog['totincomplete_pass']
                gameLog['totinterception'] = homeInt = homeInt + gameLog['interception']  
            else:
                gameLog['totHit'] = awayHit = awayHit + gameLog['qb_hit']
                gameLog['totfirst_down_rush'] = awayRush = awayRush + gameLog['totfirst_down_rush']
                gameLog['totfirst_down_pass'] = awayPass = awayPass + gameLog['totfirst_down_pass']
                gameLog['totincomplete_pass'] = awayIncomplete = awayIncomplete + gameLog['totincomplete_pass']
                gameLog['totinterception'] = awayInt = awayInt + gameLog['interception']
                for team in timeOuts:
                    if gameLog[team] == 1:
                        if team == 'TO'+posteam:
                            if gameLog['posteam_type'] == 1:
                                gameLog['home_timeouts_remaining'] -= 1
                            else:
                                gameLog['away_timeouts_remaining'] -= 1
            gameLog['ydstogo'] -= gameLog['yards_gained']
            if gameLog['ydstogo'] <= 0:
                gameLog['down'] = 1
                if gameLog['yardline_100'] <= 10:
                    gameLog['ydstogo'] = gameLog['yardline_100']
                else:
                    gameLog['ydstogo'] = 10
            else:
                gameLog['down'] +=1
            if gameLog['pass_touchdown']==1 or gameLog['rush_touchdown']==1:
                gameLog['posteam_score'] += 6
                if gameLog['posteam_type']==1:
                    gameLog['total_home_score'] += 6
                else:
                    gameLog['total_away_score'] += 6
                    
                gameLog['score_differential']=gameLog['posteam_score']-gameLog['defteam_score']
                
            if gameLog['extra_point_result'] == 1:
                gameLog['posteam_score'] += 1
                if gameLog['posteam_type']==1:
                    gameLog['total_home_score'] += 1
                else:
                    gameLog['total_away_score'] += 1
                gameLog['score_differential']=gameLog['posteam_score']-gameLog['defteam_score']
                gameLog['yard_line_100'] = 75
            if gameLog['two_point_conv_result'] == 1:
                gameLog['posteam_score'] += 2
                if gameLog['posteam_type']==1:
                    gameLog['total_home_score'] += 2
                else:
                    gameLog['total_away_score'] += 2
                gameLog['score_differential']=gameLog['posteam_score']-gameLog['defteam_score']
                gameLog['yard_line_100'] = 75
                
            if gameLog['interception'] == 1 or gameLog['punt_attempt']==1 or gameLog['fumble_lost']==1 or gameLog['down']==5:
                gameLog['drive']+=1
                saveTeam = posteam
                gameLog['P' + defteam.astype(str)] = 1#Remove Previous posteam, and add new one
                gameLog['P' + posteam.astype(str)] = 0
                gameLog['D' + posteam.astype(str)] = 1
                gameLog['D' + defteam.astype(str)] = 0
                posteam = defteam
                defteam = saveTeam
                gameLog['posteam_type']= (gameLog['posteam_type']+1)%2
                    
                 
            prediction = model.predict(gameLog).flatten()
            gameLog = simSave(gameLog)
            
            transfer = gameLog['pass_attempt', 'complete_pass', 'fumble', 'fumble_lost',
                                    'qb_hit', 'interception', 'extra_point_attempt', 
                                    'extra_point_result', 'rush_attempt', 'complete_pass',
                                    'pass_touchdown', 'yards_gained', 'rush_touchdown',
                                    'sack', 'pass_attempt', 'field_goal_attempt', 'field_goal_result',
                                    'kick_distance']
            transfer['Year'] = year
            transfer['game_id'] = game_id
                
            transfer['posteam'] = posteam
            transfer['defteam'] = defteam
            for i in range(len(throw)):
                if gameLog['Pass2' + i.astype(str)] == 1:
                    transfer['passer_player_id'] = i
                    break
            for i in range(len(rusher)):
                if gameLog['Rush' + i.astype(str)] == 1:
                    transfer['rusher_player_id'] = i
                    break
            for i in range(len(receiver)):
                if gameLog['Rec' + i.astype(str)] == 1:
                    transfer['receiver_player_id'] = i
                    break
            for i in range(len(kicker)):
                if gameLog['Kick' + i.astype(str)] == 1:
                    transfer['kicker_player_id'] = i
                    break
            try:
                saveGame = pd.concat([saveGame, transfer], axis=0, ignore_index=True)
            except NameError:#saveGame doesn't exist
                saveGame = transfer
                
                
            #list(prediction)[0
                
                
    saveGame.to_csv('gameSave.csv')
     