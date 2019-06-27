import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

game = pd.read_csv("testLast2.csv", low_memory = False)

game = game.loc[(game.SType == 'Regular') & (game.Week == 1) & (game.game_id == 2018090600)]
PO = game.loc[(game.posteam == 'PHI') ]
AO = game.loc[(game.posteam == 'ATL') ]

forcedFumble = PO['forced_fumble_player_1_player_name'].unique()
fumbleRecovery = PO['fumble_recovery_1_player_name'].unique()
fumble = PO['fumbled_1_player_name'].unique()
kicker = PO['kicker_player_name'].unique()
kReturner = PO['kickoff_returner_player_name'].unique()
pDefense = PO['pass_defense_1_player_name'].unique()
passer = PO['passer_player_name'].unique()
pReturner = PO['punt_returner_player_name'].unique()
punter = PO['punter_player_name'].unique()
qbHit = PO['qb_hit_1_player_name'].unique()
receiver = PO['receiver_player_name'].unique()
rusher = PO['rusher_player_name'].unique()
soloTackle = PO['solo_tackle_1_player_name'].unique()
tackle4Loss = PO['tackle_for_loss_1_player_name'].unique()
assistTackle = PO['assist_tackle_1_player_name'].unique()
interception = PO['interception_player_name'].unique()

players = np.concatenate((forcedFumble, fumbleRecovery, fumble, kicker, kReturner, pDefense, passer,
            pReturner, punter, qbHit, fumbleRecovery, receiver, soloTackle, tackle4Loss, assistTackle,
            interception))
Pplayer = np.unique(str(players).split())
i = 0
delete = []
for x in np.nditer(Pplayer):
    Pplayer[i] = str(x).replace("'", '')
    print(Pplayer[i])
    if('0' in str(x)) or (']' in str(x)):
        delete.append(i)
    i+=1
Pplayer = np.delete(Pplayer, delete)




forcedFumble = AO['forced_fumble_player_1_player_name'].unique()
fumbleRecovery = AO['fumble_recovery_1_player_name'].unique()
fumble = AO['fumbled_1_player_name'].unique()
kicker = AO['kicker_player_name'].unique()
kReturner = AO['kickoff_returner_player_name'].unique()
pDefense = AO['pass_defense_1_player_name'].unique()
passer = AO['passer_player_name'].unique()
pReturner = AO['punt_returner_player_name'].unique()
punter = AO['punter_player_name'].unique()
qbHit = AO['qb_hit_1_player_name'].unique()
receiver = AO['receiver_player_name'].unique()
rusher = AO['rusher_player_name'].unique()
soloTackle = AO['solo_tackle_1_player_name'].unique()
tackle4Loss = AO['tackle_for_loss_1_player_name'].unique()
assistTackle = AO['assist_tackle_1_player_name'].unique()
interception = AO['interception_player_name'].unique()

players = np.concatenate((forcedFumble, fumbleRecovery, fumble, kicker, kReturner, pDefense, passer,
            pReturner, punter, qbHit, fumbleRecovery, receiver, soloTackle, tackle4Loss, assistTackle,
            interception))
Aplayer = np.unique(str(players).split())

i = 0
delete = []
for x in np.nditer(Aplayer):
    Aplayer[i] = str(x).replace("'", '')
    print(Aplayer[i])
    if('0' in str(x)) or (']' in str(x)):
        delete.append(i)
    i+=1
Aplayer = np.delete(Aplayer, delete)
#print(Aplayer)

#print(type(interception))

game = pd.read_csv("testLast2.csv", low_memory = False)
game = game.loc[(game.SType == 'Pre') | (game.Year != 2018)]
print(len(game))
Pgame = game.loc[(game.forced_fumble_player_1_player_name.isin(Pplayer)) |
        (game['fumble_recovery_1_player_name'].isin(Pplayer)) |
        (game['fumbled_1_player_name'].isin(Pplayer)) |
        (game['kicker_player_name'].isin(Pplayer)) |
        (game['kickoff_returner_player_name'].isin(Pplayer)) |
        (game['pass_defense_1_player_name'].isin(Pplayer)) |
        (game['passer_player_name'].isin(Pplayer)) |
        (game['punt_returner_player_name'].isin(Pplayer)) |
        (game['punter_player_name'].isin(Pplayer)) |
        (game['qb_hit_1_player_name'].isin(Pplayer)) |
        (game['receiver_player_name'].isin(Pplayer)) |
        (game['rusher_player_name'].isin(Pplayer)) |
        (game['solo_tackle_1_player_name'].isin(Pplayer)) |
        (game['tackle_for_loss_1_player_name'].isin(Pplayer)) |
        (game['assist_tackle_1_player_name'].isin(Pplayer)) |
        (game['interception_player_name'].isin(Pplayer))]
        
        
print(len(Pgame))

Agame = game.loc[(game.forced_fumble_player_1_player_name.isin(Aplayer)) |
        (game['fumble_recovery_1_player_name'].isin(Aplayer)) |
        (game['fumbled_1_player_name'].isin(Aplayer)) |
        (game['kicker_player_name'].isin(Aplayer)) |
        (game['kickoff_returner_player_name'].isin(Aplayer)) |
        (game['pass_defense_1_player_name'].isin(Aplayer)) |
        (game['passer_player_name'].isin(Aplayer)) |
        (game['punt_returner_player_name'].isin(Aplayer)) |
        (game['punter_player_name'].isin(Aplayer)) |
        (game['qb_hit_1_player_name'].isin(Aplayer)) |
        (game['receiver_player_name'].isin(Aplayer)) |
        (game['rusher_player_name'].isin(Aplayer)) |
        (game['solo_tackle_1_player_name'].isin(Aplayer)) |
        (game['tackle_for_loss_1_player_name'].isin(Aplayer)) |
        (game['assist_tackle_1_player_name'].isin(Aplayer)) |
        (game['interception_player_name'].isin(Aplayer))]
        
        
print(len(Agame))

Pgame.