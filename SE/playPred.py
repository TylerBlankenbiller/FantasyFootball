#Predict what play call will be made using preplay NFL data
#Program Starts at line 159, couple of functions
#are declared above it

from __future__ import absolute_import, division, print_function

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc


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
    return train_stats

#with tf.device("/device:GPU:0"):
#running on my laptop, so had to switch GPU with something
if 1 == 1:
    tf.enable_eager_execution()

    print(tf.__version__)
    
    #load data
    training_df: pd.DataFrame = pd.read_csv("playType.csv", low_memory=False)
    training_df.loc[training_df.WTemp == '39/53', 'WTemp'] = '46'
    
    
    #Numeric Value for each play type
    #Pass       0
    #Runs       1
    #Field Goal 2
    #Punt       3
    training_df['play'] = 0
    training_df.loc[training_df.pass_attempt == 1, 'play'] = 0
    training_df.loc[training_df.rush_attempt == 1, 'play'] = 1
    training_df.loc[training_df.field_goal_attempt == 1, 'play'] = 2
    training_df.loc[training_df.punt_attempt == 1, 'play'] = 3
    
    
    #Change String stats to dummy columns
    training_df = clean(training_df)
    #Throw Out stats that are 'illegal'
    training_df = throwOut(training_df)

    #Change Type of Dataframe to type float
    dataset = training_df.copy()
    del training_df
    print(dataset.tail())
    gc.collect()
    print(dataset)
    dataset = dataset.astype(float)

    #Put play types into their own data frames
    #This helps ensure our test set is evenly distributed of each kind of play
    passer = dataset.loc[dataset.pass_attempt == 1]
    passer = passer.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])
    train_pass = passer.sample(frac=0.9,random_state=0)
    test_pass = passer.drop(train_pass.index)
    run = dataset.loc[dataset.rush_attempt == 1]
    run = run.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])
    train_run = run.sample(frac=0.9,random_state=0)
    test_run = run.drop(train_run.index)
    goal = dataset.loc[dataset.field_goal_attempt == 1]
    goal = goal.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])
    train_goal = goal.sample(frac=0.9,random_state=0)
    test_goal = goal.drop(train_goal.index)
    punt = dataset.loc[dataset.punt_attempt == 1]
    punt = punt.drop(columns=['punt_attempt', 'field_goal_attempt', 'pass_attempt', 'rush_attempt'])
    train_punt = punt.sample(frac=0.9,random_state=0)
    test_punt = punt.drop(train_punt.index)
    
    #Every Printed Length should be the Same
    print('Length of each test dataset of each type of play')
    print(len(test_punt))
    print(len(test_goal))
    print(len(test_run))
    print(len(test_pass))

    #combine these dataframes into a train and test one
    train_dataset = pd.concat([train_pass, train_run, train_punt, train_goal])
    test_dataset = pd.concat([test_pass, test_run, test_punt, test_goal])
    print(len(test_dataset))

    
    del dataset
    gc.collect()
    train_stats = train_dataset.describe()
    train_stats = train_stats.pop('play')
    train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset
    print(len(train_stats))
    print(len(train_labels))
    print("NEW STUFFF!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    test_labels = test_dataset
    plotters = test_dataset['play']
    train_labels = train_dataset.pop('play')
    test_labels = test_labels.pop('play')
    print(train_labels)
    print("Train Labels!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    #normalize Data
    def norm(x):
        return (x - train_dataset['mean']/train_dataset['std'])
        
    #train_dataset = norm(train_dataset)
    #test_dataset = norm(test_dataset)


    
    def build_model():
        model = keras.Sequential([
            keras.layers.Dense(800, activation='relu', input_shape=[len(train_dataset.keys())]),
            keras.layers.Dense(400, activation='relu'),
            keras.layers.Dense(4, activation=tf.nn.softmax)])

        optimizer = tf.keras.optimizers.RMSprop(0.1)

        model.compile(loss='mse',
                    optimizer='adam',
                    metrics=['mae', 'mse'])
        return model
    
    #Build Model
    model = build_model()

    model.summary()

    #example_batch = train_dataset[:10]
    #example_result = model.predict(example_batch)

    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: 
                print('')
            print('.', end='')

    EPOCHS = 5#22
    
    history = model.fit(train_dataset, train_labels, 
        epochs=EPOCHS, batch_size=32, validation_split=0.1, 
        verbose=0, callbacks=[PrintDot()])

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
        
    #Check accuracy
    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)

    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    test_predictions = model.predict(test_dataset).flatten()
    print(test_predictions[5])
    
    plot_history(history)
        
    s = pd.Series()
    for i in range(len(test_predictions)):
        s = s.set_value(i, np.argmax(test_predictions[i]))

    #Plot Graph of Predictions VS Actual Values
    #Vertical Line is a correct prediction
    plt.scatter(plotters, s) 
    plt.xlabel('True Values [Play Type]')
    plt.ylabel('Predictions [Play Type]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([0, 200], [0, 200])
    plt.show()
        