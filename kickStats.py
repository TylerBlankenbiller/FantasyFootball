import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def throws(x):
    d = {}
    d['short'] = x['short'].sum()
    d['med'] = x['med'].sum()
    d['long'] = x['long'].sum()
    d['attempt'] = x['attempt'].sum()
    d['longest'] = x['longest'].max()
    d['name'] = x['name'].unique()
    d['name'] = d['name'][0]
    d['id'] = x['id'].unique()
    d['id'] = d['id'][0]
    d['attempt'] = d['attempt'].sum()
    return pd.Series(d, index=['short', 'med', 'long', 'longest', 'attempt', 'name', 'id'])
    
game = pd.read_csv("testLast3.csv", low_memory = False)
kick = pd.read_csv("allActualStats.csv", low_memory = False)

game['percent'] = 0
game['longest'] = 0

kick = kick.groupby(['name', 'id']).apply(throws)

kick.to_csv('temp.csv', index=False)