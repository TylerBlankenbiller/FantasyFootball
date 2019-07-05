from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def build_model():
    model = keras.Sequential([
      layers.Dense(180, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
      layers.Dense(80, activation=tf.nn.relu),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.00001)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


if 1==1:
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
                    'yards_after_catch', 'yards_gained', 'duration'])
        return training_df
    

    raw_dataset = pd.read_csv('testLast2.csv', low_memory=False)
    raw_dataset = raw_dataset.loc[(raw_dataset.field_goal_attempt == 1)]
    raw_dataset.loc[raw_dataset.field_goal_result=='made', 'field_goal_result']='1'
    raw_dataset.loc[raw_dataset.field_goal_result=='missed', 'field_goal_result']='-1'
    raw_dataset.loc[raw_dataset.field_goal_result=='blocked', 'field_goal_result']='-1'
    raw_dataset = kicker(raw_dataset)
    dataset = raw_dataset.copy()
    dataset.tail()
    dataset.to_csv('wtf.csv', index=False)
    dataset = pd.read_csv('wtf.csv', low_memory=False)
    
    

    
    if 1==1:
        train_dataset = dataset.sample(frac=0.8,random_state=0)
        
        test_dataset = dataset.drop(train_dataset.index)

        #sns.pairplot(train_dataset[["Btomorrow", "B1", "BHigh", "BUpStreak"]], diag_kind="kde")

        train_stats = train_dataset.describe()
        train_stats.pop('field_goal_result')
        train_stats = train_stats.transpose()
        print(train_stats)

        train_labels = train_dataset.pop('field_goal_result')
        test_labels = test_dataset.pop('field_goal_result')


        normed_train_data = train_dataset
        normed_test_data = test_dataset
        
        print(normed_test_data)

        model = build_model()

        model.summary()

        example_batch = normed_train_data[:10]
        example_result = model.predict(example_batch)
        print(example_result)

        EPOCHS = 1000

        model = build_model()

        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                            validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])


        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

        print("Testing set Mean Abs Error: {:5.2f} B".format(mae))


        test_predictions = model.predict(normed_test_data)
        
        
        print(type(test_labels))
        test_labels = test_labels.values.reshape(len(test_labels),1)
        
        correct = 0
        incorrect = 0
        print(test_labels.shape)
        for i in range(len(test_predictions)):
            if(test_predictions[i] < 0):
                test_predictions[i] = -1
            else:
                test_predictions[i] = 1
            if(test_predictions[i] == test_labels[i]):
                correct += 1
            else:
                incorrect += 1

        print(correct)
        print(incorrect)
        print(correct/(incorrect+correct))
