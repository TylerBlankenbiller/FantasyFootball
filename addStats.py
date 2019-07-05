import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

game = pd.read_csv("testLast3.csv", low_memory = False)
kick = pd.read_csv("temp.csv", low_memory = False)

game = pd.concat([game, kick], axis=1, sort=False)

game.to_csv('testLast4.csv', index=False)