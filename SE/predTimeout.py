    
from __future__ import absolute_import, division, print_function

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc

def team2num(training_df, cat, *teams):
    training_df[cat] = training_df[cat].map({'0':0,'PHI': 1, 'WAS':2,
                        'DAL':3,'NYG':4,'CHI':5,'DET':6,'GB':7,
                        'MIN':8,'ATL':9,'CAR':10,'NO':11,'TB':12,
                        'ARI':13,'LA':14,'SF':15,'SEA':16,'BUF':17,
                        'MIA':18,'NE':19,'NYJ':20,'BAL':21,'CIN':22,
                        'CLE':23,'PIT':24,'HOU':25,'IND':26,'JAX':27,
                        'TEN':28,'DEN':29,'KC':30,'LAC':31,'OAK':32})
    #training_df[cat] = training_df[cat].astype(int)
    return training_df

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
    #training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
    training_df['td_team'] = 'TD' + training_df['td_team'].astype(str)
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
    training_df['fumbled_1_player_id'] = 'fum' + training_df['fumbled_1_player_id'].astype(str)

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
    training_df = training_df.drop(columns=['td_team'])
    timeOuts = list(pd.get_dummies(training_df['timeout_team']).columns.values)#############################################3
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)
    #training_df = training_df.drop(columns=['timeout_team'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['two_point_conv_result'])], axis=1)
    training_df = training_df.drop(columns=['two_point_conv_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['extra_point_result'])], axis=1)
    training_df = training_df.drop(columns=['extra_point_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['field_goal_result'])], axis=1)
    training_df = training_df.drop(columns=['field_goal_result'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['run_gap'])], axis=1)
    training_df = training_df.drop(columns=['run_gap'])
    #training_df = pd.concat([training_df, pd.get_dummies(training_df['run_location'])], axis=1)
    training_df = training_df.drop(columns=['run_location'])
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
    training_df = training_df.drop(columns=['goal_to_go'])
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
        #elif c == 'timeout':
        #    train_stats = train_stats.drop(c, axis=1)
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
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
    biases = tf.Variable(tf.zeros([200]), name="biases")
    #Test below, never tried
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config, ...)


    tf.enable_eager_execution()

    print(tf.__version__)
    teams = ['0', 'PHI', 'NYG', 'DAL', 'WAS', 'ARI', 'SEA', 'SF', 'LA',
                'ATL', 'CAR', 'NO', 'TB', 'CHI', 'DET', 'GB', 'MIN',
                'BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT',
                'HOU', 'IND', 'JAX', 'TEN', 'DEN', 'KC', 'LAC', 'OAK']
    
    #test.csv!!!!!!!!!!!!!!!!!!!!!!!!!!!@#@!#@!#@!#@!#!!@predict
    training_df: pd.DataFrame = pd.read_csv("predict.csv", low_memory=False, index_col=0)
    #training_df = training_df.loc[training_df['posteam'] != 0]
    training_df = training_df.drop(columns=['game_id', 'THRpasser_player_id', 'RECreceiver_player_id', 'RSHrusher_player_id', 'KICKkicker_player_id'])
    training_df['timeout_pos'] = 0
    training_df.loc[training_df.timeout_team == training_df.posteam, 'timeout_pos'] = 1
    training_df.loc[(training_df.timeout_team == training_df.defteam), 'timeout_pos'] = 2
    training_df = clean(training_df)
    #####training_df = team2num(training_df, 'timeout_team', teams)
    #training_df = training_df.loc[training_df.timeout_team != 0]
    training_df = training_df.drop(columns=['timeout_team'])
    training_df = throwOut(training_df)
    #training_df.to_csv('wrong.csv')

    
    

    ################################################################################################
    # Create some variables.
    #v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
    #v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

    #inc_v1 = v1.assign(v1+1)
    #dec_v2 = v2.assign(v2-1)

    # Add an op to initialize the variables.
    #init_op = tf.global_variables_initializer()

    ## Add ops to save and restore all the variables.
    #saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    #with tf.Session() as sess:
    #    sess.run(init_op)
    #    # Do some work with the model.
    #    inc_v1.op.run()
    #    dec_v2.op.run()
    #    # Save the variables to disk.
    #    save_path = saver.save(sess, "/tmp/model.ckpt")
    #    print("Model saved in path: %s" % save_path)
    #    ###########################################################################################


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
    #sns.pairplot(train_dataset[["game_seconds_remaining", "drive", "ydstogo", "yards_gained"]], diag_kind="kde")
    #plt.show()
    print("gooder")
    train_stats = train_dataset.describe()
    print("goodest")
    
    
    
    #train_stats = throwOut(train_stats)   
    train_stats = train_stats.pop('timeout_pos')
    
    
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset.pop('timeout_pos')
    test_labels = test_dataset.pop('timeout_pos')

    def norm(x):
        return (x - train_stats['mean']/train_stats['std'])
    
    train_dataset = norm(train_dataset)
    test_dataset = norm(test_dataset)

    #normed_train_data = norm(train_dataset)
    #normed_test_data = norm(test_dataset)
    print('Good')
    def build_model():
        model = keras.Sequential([
            layers.Dense(2048, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
            layers.Dense(1024, activation=tf.nn.relu),
            layers.Dense(3, activation=tf.nn.softmax)])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model
        
    model = build_model()

    model.summary()

    example_batch = train_dataset[:10]
    example_result = model.predict(example_batch)

    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: 
                print('')
            print('.', end='')

    EPOCHS = 10

    history = model.fit(
        train_dataset, train_labels,
        epochs=EPOCHS,
        callbacks=[PrintDot()])
        
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [timeout_team]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label = 'Val Error')
        plt.ylim([-100,1000])
        plt.legend()
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$timeout_team^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label = 'Val Error')
        plt.ylim([-100,2000])
        plt.legend()
        plt.show()


    #plot_history(history)

    test_loss, test_acc = model.evaluate(test_dataset, test_labels)

    print('Test accuracy:', test_acc)

    test_predictions = model.predict(test_dataset)
    print(test_predictions[20])
    
    s = pd.Series()
    for i in range(len(test_predictions)):
        s = s.set_value(i, np.argmax(test_predictions[i]))
    print(s)
    print("S")
        
    count = 0
    total = 0
    for i in range(len(test_predictions)):
        total+=1
        if s.index[i] != test_dataset.index[i]:
            count += 1
            #print(s.index[i])
            #print(test_dataset.index[i])
            
    print(count)
    print(total)
    
    plt.scatter(test_labels, s) 
    plt.xlabel('True Values [Timeout Team]')
    plt.ylabel('Predictions [Timout Team]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([0, 200], [0, 200])
    plt.show()
    
    
    
