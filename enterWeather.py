import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getMascot(x):
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
        for i in range(len(value)):
            if str(x) == value[i]:
                return key
    return 'NA'

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
    

num = 'test'
for idx in range(7):
    idx+=2
    num = str(idx + 10)
    print(num)
        
    data = pd.read_csv("reg_up_20"+num+".csv", low_memory=False)
    weather = pd.read_csv("weather.csv", low_memory=False)
    
    data['home_mascot'] = np.vectorize(getMascot)(data['home_team'])
    data['away_mascot'] = np.vectorize(getMascot)(data['away_team'])
    
    weather['away_team'] = np.vectorize(getCity)(weather['away_mascot'], weather['Year'])
    weather['home_team'] = np.vectorize(getCity)(weather['home_mascot'], weather['Year'])
    
    
    new = weather.Wind.str.split('m',1, expand = True)
    weather['WSpeed'] = new[0]
    weather['WDirection'] = new[1]
    new = weather.Weather.str.split('f',1, expand = True)
    weather['WTemp'] = new[0]
    weather['Weather'] = new[1]
    
    result = pd.merge(data, weather, how='left', on=['away_team', 'home_team', 'Year', 'Week'])
    
    result = result.reset_index(drop=True)
    #data['Weather'] = 'NA'
    #for i in range(len(data)):
    #    data['Weather'] = 'NA'
    #    data['Wind'] = 'NA'
    #    for j in range(len(weather)):
    #        if (data['away_team'][i] == weather['away_team'][j]) & (data['home_team'][i] == weather['home_team'][j]) & (data['Week'][i] == weather['Week'][j]) & (data['Year'][i] == weather['Year'][j]):
    #            data['Weather'][i] = weather['Weather'][j]
    #            data['Wind'][i] = weather['Wind'][j]
    
    #new = result.Wind.str.split('m',1, expand = True)
    #result['WSpeed'] = new[0]
    #result['WDirection'] = new[1]
    ##new = result.Weather.str.split('f',1, expand = True)
    #result['WTemp'] = new[0]
    #result['Weather'] = new[1]
    
    result.to_csv('reg_comb_20'+num+'.csv')