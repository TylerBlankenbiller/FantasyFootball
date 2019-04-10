    
from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc

def throwOut(train_stats):
    for c in train_stats.columns:
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
        elif c == 'Pre':
            train_stats = train_stats.drop(c, axis=1)
        elif c == 'Regular':
            train_stats = train_stats.drop(c, axis=1)
    return train_stats

tf.enable_eager_execution()

with tf.device("/device:GPU:0"):
    #Test below, never tried
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config, ...)


    tf.enable_eager_execution()

    print(tf.__version__)

    training_df: pd.DataFrame = pd.read_csv("test.csv", low_memory=False, index_col=0)


    
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
    training_df['posteam_type'] = 'postype' + training_df['AOffense'].astype(str)
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['kicker_player_id'])], axis=1)
    training_df = training_df.drop(columns=['kicker_player_id'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['rusher_player_id'])], axis=1)
    training_df = training_df.drop(columns=['rusher_player_id'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['receiver_player_id'])], axis=1)
    training_df = training_df.drop(columns=['receiver_player_id'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['passer_player_id'])], axis=1)
    training_df = training_df.drop(columns=['passer_player_id'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['td_team'])], axis=1)
    training_df = training_df.drop(columns=['td_team'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['timeout_team'])], axis=1)
    training_df = training_df.drop(columns=['timeout_team'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['two_point_conv_result'])], axis=1)
    training_df = training_df.drop(columns=['two_point_conv_result'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['extra_point_result'])], axis=1)
    training_df = training_df.drop(columns=['extra_point_result'])
    training_df = pd.concat([training_df, pd.get_dummies(training_df['field_goal_result'])], axis=1)
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
    training_df = pd.concat([training_df, pd.get_dummies(training_df['fumbled_1_player_id'])], axis=1)
    training_df = training_df.drop(columns=['fumbled_1_player_id'])

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
    
    
    
    train_stats = throwOut(train_stats)   

    
    
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset
    test_labels = train_dataset
    train_labels = throwOut(train_labels)   
    test_labels = throwOut(test_labels)   

    #train_labels = train_dataset.pop(Aoff)

    #def norm(x):
        #return (x - x['yards_gained'].sum()/x['yards_gained'].count() / train_stats['yards_gained'].std()

    #normed_train_data = norm(train_dataset)
    #normed_test_data = norm(test_dataset)
    print('Good')

    def build_model():
        model = keras.Sequential([
            layers.Dense(512, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(1)])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
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

    EPOCHS = 30

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