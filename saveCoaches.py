#Add the coaches to the csv with everything else
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


found = False
coach = pd.read_csv("coaches2.csv", low_memory = False)

for idx in range(7):
    idx+=2
    num = str(idx + 10)
        
    data = pd.read_csv("reg_comb_20"+num+".csv", low_memory=False)

    print("year")
    print(num)
    
    merged = pd.merge(data, coach, left_on=['home_mascot_x', 'Week', 'Year'], right_on=['Team', 'Week', 'Year'], how='left')
    merged.rename(columns={'Coach':'HCoach', 'Defense':'HDefense', 'Offense':'HOffense'}, inplace=True)
    print('a')
    merged = pd.merge(merged, coach, left_on=['away_mascot_x', 'Week', 'Year'], right_on=['Team', 'Week', 'Year'], how='left')
    merged.rename(columns={'Coach':'ACoach', 'Defense':'ADefense', 'Offense':'AOffense', 'home_mascot_y': 'home_mascot', 'away_mascot_y': 'away_mascot'}, inplace=True)
    print('b')
    merged = merged.drop(columns=['home_mascot_x', 'away_mascot_x', 'Team_x', 'Team_y', 'Unnamed: 0', 'Unnamed: 0.1'])
    merged.to_csv('almost_20'+num+'.csv')