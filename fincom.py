import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#d2018 = pd.read_csv("almost_2018.csv", low_memory=False)
d2017 = pd.read_csv("almost_2017.csv", low_memory=False)
d2016 = pd.read_csv("almost_2016.csv", low_memory=False)
d2015 = pd.read_csv("almost_2015.csv", low_memory=False)
d2014 = pd.read_csv("almost_2014.csv", low_memory=False)
d2013 = pd.read_csv("almost_2013.csv", low_memory=False)
d2012 = pd.read_csv("almost_2012.csv", low_memory=False)


almost = pd.concat([d2017, d2016, d2015, d2014, d2013, d2012], axis=0, ignore_index=True,sort=False)
text_file = open("Games.txt", "w")
text_file.write(almost.game_id.unique())
text_file.close()
#print(almost.game_id.unique())
#s.cumsum()
#USE THAT^^^

#almost.to_csv('final.csv')
