from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

def my_input_fn():
    examples = tf.learn.graph_io.read_batch_examples(
        'finish2012_2018.csv', 32, tf.TextLineReader)
    header = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex',
        'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    record_defaults = [[1], [1], [1], [''], [''], [1], [1], [1], [''], [1.0], [''], ['']]
    cols = tf.decode_csv(examples, record_defaults=record_defaults)
    features = zip(header, cols)
    target = features.pop('Survived')
    return features, target