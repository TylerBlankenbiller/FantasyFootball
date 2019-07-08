import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

game = pd.read_csv("testLast3.csv", low_memory = False)
kick = pd.read_csv("temp.csv", low_memory = False)

game = game.merge(kick, how = 'inner', on = ['kicker_player_id'])
game['attempt'] += 1
game['acc'] = (game['short']+game['med']+game['long'])/game['attempt']
game['attempt'] -= 1

game.to_csv('testLast4.csv', index=False)