import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

num = 'test'
coach = pd.read_csv("coaches.csv", low_memory=False)
coach = coach.loc[coach['Year'] >= 2012]
#coach = coach.fillna(method='ffill')
coach = coach.iloc[::-1]
coach['Week'] = 0
coach2 = coach.copy()
#coach = coach.groupby(coach.columns.tolist(),as_index=False).size()
for index, season in coach.iterrows():
    for week in range(season['G']):
        game = week +1
        #print(season)
        sea = season.copy()
        coach2 = coach2.append(sea, ignore_index=False)
        coach2.reindex()
        #print(coach2.loc[index, 'Week'])
        coach2.at[index, 'Week'] = game
        #print(season)
        
coach2.to_csv('coaches2.csv') 

#for idx in range(7):
#    idx+=2
#    num = str(idx + 10)
#    print(num)
#        
#    data = pd.read_csv("reg_comb_20"+num+".csv", low_memory=False)
    
    