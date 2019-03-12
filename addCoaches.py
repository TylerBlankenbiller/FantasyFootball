#Reformat the coach info
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

num = 'test'
coach = pd.read_csv("coaches.csv", low_memory=False)
coach['Year'] = coach['Year'].fillna(method='ffill')
coach = coach.loc[coach['Year'] >= 2012]
coach = coach.iloc[::-1]
coach.to_csv('coaches3.csv') 
coach['Week'] = 0
coach2 = pd.DataFrame()
sYear = []
i = 0
weeks=0
#coach = coach.groupby(coach.columns.tolist(),as_index=False).size()
for index, season in coach.iterrows():
    sYear.append(season)
    #weeks = 0
    #print("!")
    
    while i < len(sYear):
        #print(sYear[i])
        weeks+=sYear[i]['G']
        #print(weeks)
        i+=1
    #print(weeks)
    if weeks == 16:
        game = 0
        i = 0
        while i < len(sYear):
            for week in range(sYear[i]['G']):
                game = game +1
                if(game == 12):
                    d = {}
                    d['Year'] = sYear[i]['Year']
                    d['Coach'] = sYear[i]['Coach']
                    d['Offense'] = sYear[i]['Offense']
                    d['Defense'] = sYear[i]['Defense']
                    d['Team'] = sYear[i]['Team']
                    d['Week'] = game
                    coach2 = coach2.append(pd.Series(d, index=['Year', 'Coach', 'Offense', 'Defense', 'Team', 'Week']), ignore_index=True)
                    game+=1
                #print(season)
                d = {}
                #print(i)
                d['Year'] = sYear[i]['Year']
                d['Coach'] = sYear[i]['Coach']
                d['Offense'] = sYear[i]['Offense']
                d['Defense'] = sYear[i]['Defense']
                d['Team'] = sYear[i]['Team']
                d['Week'] = game
                coach2 = coach2.append(pd.Series(d, index=['Year', 'Coach', 'Offense', 'Defense', 'Team', 'Week']), ignore_index=True)
            i+=1
        i = 0
        sYear = []
        weeks = 0

        
coach2.to_csv('coaches2.csv') 
    
    