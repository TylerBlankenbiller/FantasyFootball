#Add the coaches to the csv with everything else
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getCity(x, y):
    teams = {   'Falcons':  ['ATL'],
                'Saints':  ['NO'],
                'Buccaneers':  ['TB'],
                'Panthers':  ['CAR'],
                'Eagles':  ['PHI'],
                'Giants':  ['NYG'],
                'Cowboys':  ['DAL'],
                'Redskins':  ['WAS'],
                'Rams':  ['LA', 'STL'],
                'Seahawks':  ['SEA'],
                '49ers':  ['SF'],
                'Cardinals':  ['ARI'],
                'Bears':  ['CHI'],
                'Packers':  ['GB'],
                'Vikings':  ['MIN'],
                'Lions':  ['DET'],
                'Patriots':  ['NE'],
                'Dolphins':  ['MIA'],
                'Jets':  ['NYJ'],
                'Bills':  ['BUF'],
                'Bengals':  ['CIN'],
                'Steelers':  ['PIT'],
                'Browns':  ['CLE'],
                'Ravens':  ['BAL'],
                'Chiefs':  ['KC'],
                'Broncos':  ['DEN'],
                'Raiders':  ['OAK'],
                'Chargers':  ['LAC', 'SD'],
                'Colts':  ['IND'],
                'Jaguars':  ['JAX'],
                'Titans':  ['TEN'],
                'Texans':  ['HOU']
            }
    for key, value in teams.items():#Check Dictionary Keys
        if str(x) == key:
            if (int(y) < 2016) & len(value) > 1:
                return value[1]
            else:
                return value[0]
    return 'NA'
   

found = False
coach = pd.read_csv("coaches2.csv", low_memory = False)
coach['Team'] = np.vectorize(getCity)(coach['Team'], coach['Year'])

for idx in range(7):
    idx+=2
    num = str(idx + 10)
        
    data = pd.read_csv("reg_up_20"+num+".csv", low_memory=False)
    data = data.loc[data['Weather'] != '']
    print("year")
    print(num)
    
    if idx == 8:
        data = data.loc[data['SType'] == 'Pre']
    
    merged = pd.merge(data, coach, left_on=['Home', 'Week', 'Year'], right_on=['Team', 'Week', 'Year'], how='left')
    merged.rename(columns={'Coach':'HCoach', 'Defense':'HDefense', 'Offense':'HOffense'}, inplace=True)
    print('a')
    merged = pd.merge(merged, coach, left_on=['Away', 'Week', 'Year'], right_on=['Team', 'Week', 'Year'], how='left')
    merged.rename(columns={'Coach':'ACoach', 'Defense':'ADefense', 'Offense':'AOffense', 'home_mascot_y': 'home_mascot', 'away_mascot_y': 'away_mascot'}, inplace=True)
    print('b')
    #merged = merged.drop(columns=['home_mascot_x', 'away_mascot_x', 'Team_x', 'Team_y', 'Unnamed: 0', 'Unnamed: 0.1'])
    merged = merged.loc[merged['ACoach'] != '']
    merged.to_csv('almost_20'+num+'.csv')