import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

d2018 = pd.read_csv("players_2018.csv", low_memory=False)
d2017 = pd.read_csv("players_2017.csv", low_memory=False)
d2016 = pd.read_csv("players_2016.csv", low_memory=False)
d2015 = pd.read_csv("players_2015.csv", low_memory=False)
d2014 = pd.read_csv("players_2014.csv", low_memory=False)
d2013 = pd.read_csv("players_2013.csv", low_memory=False)
d2012 = pd.read_csv("players_2012.csv", low_memory=False)
d2011 = pd.read_csv("players_2011.csv", low_memory=False)
d2010 = pd.read_csv("players_2010.csv", low_memory=False)
d2009 = pd.read_csv("players_2009.csv", low_memory=False)

players = pd.concat([d2018, d2017, d2016, d2015, d2014, d2013, d2012, d2011, d2010, d2009], axis=0, ignore_index=True)

players.to_csv('players_2009_2018.csv')
out = players.to_json(orient='records')[1:-1].replace('},{', '} {')
with open('players2009_2018.txt', 'w') as f:
    f.write(out)