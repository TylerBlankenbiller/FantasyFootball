import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

game = pd.read_csv("testLast2.csv", low_memory = False)
game['timeout'] = game['timeout'].shift(-1)
game['timeout_team'] = game['timeout_team'].shift(-1)
game.to_csv('testLast3.csv')