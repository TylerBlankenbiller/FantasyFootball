import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

game = pd.read_csv("testLast2.csv", low_memory = False)

for index, row in game.iterrows():
    if(row['timeout_team'] == row['posteam']):
        game['timeout_team'][index] = '-1'
    else:
        game['timeout_team'][index] = '1'
game.to_csv("testLast2.csv", index=False)