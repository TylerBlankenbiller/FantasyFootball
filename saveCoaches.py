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
    
    data['HomeHCoach'] = 'NA'
    data['HomeDCoach'] = 'NA'
    data['HomeOCoach'] = 'NA'
    data['AwayHCoach'] = 'NA'
    data['AwayDCoach'] = 'NA'
    data['AwayOCoach'] = 'NA'
    print("year")
    print(num)
    
    for index, play in data.iterrows():
        for ind, week in coach.iterrows():
            
            if(play['Week'] == week['Week']) & (play['Year'] == week['Year']):
                one = str(play['home_mascot_x'])
                two = str(week['Team'])
                print(index)
                #print(week['Team'])
                if(one == two):
                    #print("Good")
                    data.loc['HomeHCoach', index] = week['Coach']
                    data.loc['HomeDCoach',index] = week['Defense']
                    data.loc['HomeOCoach',index] = week['Offense']
                elif(str(play['away_mascot_x']) == str(week['Team'])):
                    data.loc['AwayHCoach',index] = week['Coach']
                    data.loc['AwayDCoach',index] = week['Defense']
                    data.loc['AwayOCoach',index] = week['Offense']
                break
    data.to_csv('reg_coaches_20'+str(num)+'.csv')
    break
    