import matplotlib.pyplot as plt
import matplotlib as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import feature_column_ops

def my_input_fn():
    examples = pd.read_csv("test.csv", low_memory=False)
    #header = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex',
    #    'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    #record_defaults = [[1], [1], [1], [''], [''], [1], [1], [1], [''], [1.0], [''], ['']]
    cols = tf.decode_csv(examples)
    features = zip(examples.columns.names, cols)
    target = features.pop('Survived')
    return features, target
    
x = my_input_fn()