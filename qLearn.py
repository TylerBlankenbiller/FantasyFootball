from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


raw_dataset = pd.read_csv("final.csv", low_memory=False)
                      
dataset = raw_dataset.copy()
dataset.tail()
print(type(dataset))
dataset.feature_column.categorical_column_with_identity(
    'home_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'away_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'posteam',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'posteam_type',
    2,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'defteam',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'side_of_field',
    33,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'game_half',
    2,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'play_type',
    6,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'pass_length',
    2,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'pass_location',
    3,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'run_location',
    3,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'run_gap',
    3,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'field_goal_result',
    2,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'extra_point_result',
    2,
    default_value=None
)dataset.feature_column.categorical_column_with_identity(
    'two_point_conv_result',
    2,
    default_value=None
)dataset.feature_column.categorical_column_with_identity(
    'timeout_team',
    32,
    default_value=None
)dataset.feature_column.categorical_column_with_identity(
    'td_team',
    32,
    default_value=None
)dataset.feature_column.categorical_column_with_identity(
    'passer_player_name',
    2,
    default_value=None
)dataset.feature_column.categorical_column_with_identity(
    'assist_tackle_1_team',
    32,
    default_value=None
)dataset.feature_column.categorical_column_with_identity(
    'solo_tackle_2_team',
    32,
    default_value=None
)dataset.feature_column.categorical_column_with_identity(
    'solo_tackle_1_team',
    32,
    default_value=None
)dataset.feature_column.categorical_column_with_identity(
    'forced_fumble_player_2_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'forced_fumble_player_1_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'assist_tackle_2_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'assist_tackle_3_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'assist_tackle_4_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'fumbled_1_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'fumbled_2_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'fumble_recovery_1_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'fumble_recovery_2_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'return_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'penalty_team',
    32,
    default_value=None
)
dataset.feature_column.categorical_column_with_identity(
    'replay_or_challenge_result',
    2,
    default_value=None
)



train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("no_huddle")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('no_huddle')
test_labels = test_dataset.pop('no_huddle')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model
  
model = build_model()

model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
      normed_train_data, train_labels,
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
    plt.ylabel('Mean Abs Error [no_huddle]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$no_huddle^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()


plot_history(history)

odel = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])