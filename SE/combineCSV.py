import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc

players = pd.read_csv("final.csv", low_memory=False)
combine = pd.read_csv('combine20.csv')

combine['Player'] = combine['Player'].str[0]+'.'+ combine['Player'].str.split(" ",1).str[1] 

#qb = combine.copy()
#qb.columns = ['QB_'+str(col) for col in qb.columns]
#players = pd.merge(players, qb,  how='left', left_on=['passer_player_name'], right_on = ['QB_Player'])
#del qb
#print("QB")

catch = combine.copy()
catch.columns = ['catch_'+str(col) for col in catch.columns]
players = pd.merge(players, catch,  how='left', left_on=['receiver_player_name'], right_on = ['catch_Player'])
del catch
print("Catach")

#run = combine.copy()
#run.columns = ['run_'+str(col) for col in run.columns]
#players = pd.merge(players, run,  how='left', left_on=['rusher_player_name'], right_on = ['run_Player'])
#del run

#print("Run")

#solo = combine.copy()
#solo.columns = ['solo_'+str(col) for col in solo.columns]
#players = pd.merge(players, solo,  how='left', left_on=['solo_tackle_1_player_name'], right_on = ['solo_Player'])
#del solo
#print("Solo")

#defe = combine.copy()
#defe.columns = ['def_'+str(col) for col in defe.columns]
#players = pd.merge(players, defe,  how='left', left_on=['pass_defense_1_player_name'], right_on = ['def_Player'])
#del defe
#print("Def")

players.to_csv('finish2012_2018.csv')

