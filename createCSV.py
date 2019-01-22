import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("reg_pbp_2017STATS.csv", low_memory=False)

players = pd.DataFrame({'Name': ['Nan'], 'Pos': ['Nan'], 'Team': ['Nan'], 'Comp': [0], 'PAtt': [0], 'PYds': [0], 'PLong': [0], 'PTD': [0], 'Int': [0],
            'Sack': [0], 'Rate': [0], 'RYds': [0], 'RTD': [0], 'Fum': [0], 'Lst': [0], 'Rec': [0], 'Tgts': [0], 'RecYds': [0],
            'RecLong': [0], 'RecTD': [0], 'RAtt': [0], 'RYds': [0], 'RLong': [0], 'RTD': [0], 'Fantasy': [0]})
#players = pd.DataFrame()
#players.append(temp)

row2 = 0
#print(data['touchdown'])
found = False  

throw = data
throw = throw.loc[throw['penalty'] == 0]
throw = throw.loc[throw['play_type'] == 'pass']
throw['tattempts'] = 1
throw['player'] = throw['passer_player_name']
throw['completions'] = throw['tattempts'] - throw['incomplete_pass']
#throw = throw.groupby(['passer_player_id', 'posteam']).sum()
throw['percent'] = throw['completions']/throw['tattempts']
throw['y/att'] = throw['yards_gained']/throw['tattempts']
#throw = throw.sort_values(['yards_gained'])
throw['passYards'] = throw['yards_gained']
#print(throw)
#print(throw['sack'])

run = data
run = run.loc[run['penalty'] == 0]
run = run.loc[run['play_type'] == 'run']
run['name'] = run['rusher_player_name']
run['rattempts'] = 1
#run = run.groupby(['rusher_player_name', 'posteam']).sum()
#run = run.sort_values(['yards_gained'])
run['runYards'] = run['yards_gained']
run['rAVG'] = run['runYards']/run['rattempts']
run['rTD'] = run['touchdown']
#print(run['rattempts'], run['runYards'], run['rAVG'], run['rTD'], run['fumble'], run['fumble_lost'])

receiver = data
receiver = receiver.loc[receiver['penalty'] == 0]
receiver = receiver.loc[receiver['play_type'] == 'pass']
receiver['name'] = receiver['receiver_player_name'] 
receiver['target'] = 1
#receiver = receiver.groupby(['receiver_player_name', 'posteam']).sum()
#receiver = receiver.sort_values(['yards_gained'])
print(receiver)




players = pd.concat([throw, run, receiver], axis=0, ignore_index=True, sort = False)
players = players.groupby(['name', 'posteam']).sum()
players = players.sort_values(['yards_gained'])
#players = receiver.groupby(['name', 'posteam']).sum()
print(players)