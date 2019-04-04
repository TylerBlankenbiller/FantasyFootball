import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calc(x):
    d = {}
    d['play_id'] = x['play_id'].average()
    d['first_down_rush'] = x[''].sum()
    d['first_down_pass'] = x['long'].sum()
    d['first_down_penalty'] = x['kick_distance'].max()
    d['incomplete_pass'] = x['attempt'].sum()
    d['PAT_Attempt'] = x['PAT_Attempt'].sum()
    d['PAT_Made'] = x['PAT_Made'].sum() 
    return pd.Series(d, index=['short', 'med', 'long', 'longest', 'attempt', 'PAT_Attempt', 'PAT_Made'])
    
def f(x):
    print('Almost')

    return x.groupby(x.ne().cumsum()).cumcount() + 1

#i = df.apply(pd.to_numeric, errors='coerce')

#d2018 = pd.read_csv("almost_2018.csv", low_memory=False)
d2017 = pd.read_csv("all_2017.csv", low_memory=False)
d2016 = pd.read_csv("all_2016.csv", low_memory=False)
d2015 = pd.read_csv("all_2015.csv", low_memory=False)
d2014 = pd.read_csv("all_2014.csv", low_memory=False)
d2013 = pd.read_csv("all_2013.csv", low_memory=False)


almost = pd.concat([d2017, d2016, d2015, d2014, d2013], axis=0, ignore_index=True,sort=False)

#games = games.apply(f)
almost['totHit'] = almost.groupby(['game_id', 'posteam'])['qb_hit'].cumsum()
almost['totfirst_down_rush'] = almost.groupby(['game_id', 'posteam'])['first_down_rush'].cumsum()
almost['totfirst_down_pass'] = almost.groupby(['game_id', 'posteam'])['first_down_pass'].cumsum()
almost['totfirst_down_penalty'] = almost.groupby(['game_id', 'posteam'])['first_down_penalty'].cumsum()
almost['totincomplete_pass'] = almost.groupby(['game_id', 'posteam'])['incomplete_pass'].cumsum()
almost['totinterception'] = almost.groupby(['game_id', 'posteam'])['interception'].cumsum()
print('Almost')
almost.to_csv('games.csv')

#gameTeam = almost.groupby(['week', 'date', 'game_id', 'name', 'game_id', 'posteam', 'defteam']).sum()
#USE THAT^^^

#almost.to_csv('final.csv')
