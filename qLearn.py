from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc

tf.enable_eager_execution()

print(tf.__version__)

training_df: pd.DataFrame = pd.read_csv("final.csv", low_memory=False)

training_df = training_df.replace({'STL':'LA'}, regex=True)
training_df = training_df.replace({'SD':'LAC'}, regex=True)
training_df = training_df.replace({'JAC':'JAX'}, regex=True)
training_df['home_team'] = 'H' + training_df['home_team'].astype(str)
training_df['away_team'] = 'A' + training_df['away_team'].astype(str)
training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
training_df['extra_point_result'] = 'Extra' + training_df['extra_point_result'].astype(str)
training_df['two_point_conv_result'] = 'Two' + training_df['two_point_conv_result'].astype(str)
training_df['play_type'] = 'Play' + training_df['play_type'].astype(str)
training_df['run_gap'] = 'Gap' + training_df['run_gap'].astype(str)
training_df['field_goal_result'] = 'Field' + training_df['field_goal_result'].astype(str)
training_df['timeout_team'] = 'TO' + training_df['timeout_team'].astype(str)
training_df['td_team'] = 'TD' + training_df['td_team'].astype(str)
training_df['passer_player_id'] = 'Pass' + training_df['passer_player_id'].astype(str)
training_df['receiver_player_id'] = 'Rec' + training_df['receiver_player_id'].astype(str)
training_df['rusher_player_id'] = 'Rush' + training_df['rusher_player_id'].astype(str)
training_df['kicker_player_id'] = 'Kick' + training_df['kicker_player_id'].astype(str)
training_df['penalty_team'] = 'PTeam' + training_df['penalty_team'].astype(str)
training_df['replay_or_challenge_result'] = 'Replay' + training_df['replay_or_challenge_result'].astype(str)
training_df['penalty_type'] = 'Pen' + training_df['penalty_type'].astype(str)
training_df['Weather'] = 'Weather' + training_df['Weather'].astype(str)
training_df['WDirection'] = training_df['WDirection'].astype(str)
training_df['HCoach'] = 'HCo' + training_df['HCoach'].astype(str)
training_df['HDefense'] = 'HDef' + training_df['HDefense'].astype(str)
training_df['ACoach'] = 'ACo' + training_df['ACoach'].astype(str)
training_df['ADefense'] = 'ADef' + training_df['ADefense'].astype(str)
training_df['AOffense'] = 'AOff' + training_df['AOffense'].astype(str)
training_df['posteam_type'] = 'postype' + training_df['AOffense'].astype(str)
#training_df[''] = '' + training_df[''].astype(str)
#training_df[''] = '' + training_df[''].astype(str)
#training_df[''] = '' + training_df[''].astype(str)
#training_df[''] = '' + training_df[''].astype(str)
#training_df[''] = '' + training_df[''].astype(str)
Aoff = list(pd.get_dummies(training_df['AOffense']).columns.values)
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
training_df = pd.concat([training_df, pd.get_dummies(training_df['penalty_type'])], axis=1)
training_df = training_df.drop(columns=['penalty_type'])
training_df = pd.concat([training_df, pd.get_dummies(training_df['replay_or_challenge_result'])], axis=1)
training_df = training_df.drop(columns=['replay_or_challenge_result'])
training_df = pd.concat([training_df, pd.get_dummies(training_df['penalty_team'])], axis=1)
training_df = training_df.drop(columns=['penalty_team'])
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
training_df = pd.concat([training_df, pd.get_dummies(training_df['play_type'])], axis=1)
training_df = training_df.drop(columns=['play_type'])
training_df = pd.concat([training_df, pd.get_dummies(training_df['defteam'])], axis=1)
training_df = training_df.drop(columns=['defteam'])
training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam'])], axis=1)
training_df = training_df.drop(columns=['posteam'])
training_df = pd.concat([training_df, pd.get_dummies(training_df['away_team'])], axis=1)
training_df = training_df.drop(columns=['away_team'])
training_df = pd.concat([training_df, pd.get_dummies(training_df['home_team'])], axis=1)
training_df = training_df.drop(columns=['home_team'])
training_df = pd.concat([training_df, pd.get_dummies(training_df['posteam_type'])], axis=1)
training_df = training_df.drop(columns=['posteam_type'])

training_df.loc[training_df.WTemp == 'DOME', 'WTemp']=70
training_df.loc[training_df.WTemp == '33/51', 'WTemp']=40
dataset = training_df.copy()
del training_df
print(dataset.tail())
gc.collect()
dataset = dataset.fillna(0)

dataset = dataset.astype(float)

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

del dataset
gc.collect()
print("good")
sns.pairplot(train_dataset[["game_seconds_remaining", "drive", "ydstogo", "ydsnet"]], diag_kind="kde")
plt.show()
print("gooder")
train_stats = train_dataset.describe()
print("goodest")
train_stats.pop("ydsnet")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop(Aoff)


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)