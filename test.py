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
    raw_dataset = pd.read_csv('wtf.csv')
    dataset = raw_dataset.copy()
    dataset.tail()
    
    

    
    if 1==1:
        train_dataset = dataset.sample(frac=0.75,random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        #sns.pairplot(train_dataset[["Btomorrow", "B1", "BHigh", "BUpStreak"]], diag_kind="kde")

        train_stats = train_dataset.describe()
        train_stats.pop("timeout_team")
        train_stats = train_stats.transpose()
        print(train_stats)

        train_labels = train_dataset.pop('timeout_team')
        test_labels = test_dataset.pop('timeout_team')


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
