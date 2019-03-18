from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.enable_eager_execution()

print(tf.__version__)

training_df: pd.DataFrame = pd.read_csv("final.csv", low_memory=False)

training_df['home_team'] = 'H' + training_df['home_team'].astype(str)
training_df['away_team'] = 'A' + training_df['away_team'].astype(str)
training_df['posteam'] = 'P' + training_df['posteam'].astype(str)
training_df['defteam'] = 'D' + training_df['defteam'].astype(str)
training_df['pass_location'] = 'Pass' + training_df['pass_location'].astype(str)
training_df['run_location'] = 'Run' + training_df['run_location'].astype(str)
training_df['extra_point_result'] = 'Extra' + training_df['extra_point_result'].astype(str)
training_df['two_point_conv_result'] = 'Two' + training_df['two_point_conv_result'].astype(str)
training_df[''] = '' + training_df[''].astype(str)
training_df[''] = '' + training_df[''].astype(str)
training_df[''] = '' + training_df[''].astype(str)
training_df[''] = '' + training_df[''].astype(str)
training_df[''] = '' + training_df[''].astype(str)
print(type(training_df))

#features = ['

#target = ['











