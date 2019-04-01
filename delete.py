import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

found = False
coach = pd.read_csv("coaches2.csv", low_memory = False)

for idx in range(7):
    idx+=2
    num = str(idx + 10)
        
    data = pd.read_csv("almost_20"+num+".csv", low_memory=False)

    print("year")
    print(num)
    
    data = data.drop(['Unnamed: 0', 'play_id', 'game_date', 'game_half', 'time', 'yrdln', 'pass_length',
        'passer_player_name', 'receiver_player_name', 'rusher_player_name', 'lateral_receiver_player_id',
        'lateral_receiver_player_name', 'lateral_rusher_player_id', 'lateral_rusher_player_name',
        'lateral_sack_player_id', 'lateral_sack_player_name', 'interception_player_id', 
        'interception_player_name', 'lateral_interception_player_id', 'lateral_interception_player_name',
        'punt_returner_player_id', 'punt_returner_player_name', 'lateral_punt_returner_player_id', 
        'lateral_punt_returner_player_name', 'kickoff_returner_player_name', 'kickoff_returner_player_id',
        'lateral_kickoff_returner_player_id', 'lateral_kickoff_returner_player_name', 'punter_player_name',
        'kicker_player_name', 'own_kickoff_recovery_player_id', 'own_kickoff_recovery_player_name',
        'blocked_player_id', 'blocked_player_name', 'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name',
        'tackle_for_loss_2_player_id', 'tackle_for_loss_2_player_name', 'qb_hit_1_player_id', 'qb_hit_1_player_name',
        'qb_hit_2_player_id', 'qb_hit_2_player_name', 'forced_fumble_player_1_team', 'forced_fumble_player_1_player_id',
        'forced_fumble_player_1_player_name', 'forced_fumble_player_2_team', 'forced_fumble_player_2_player_id',
        'forced_fumble_player_2_player_name', 'solo_tackle_1_team', 'solo_tackle_2_team', 'solo_tackle_1_player_id',
        'solo_tackle_2_player_id', 'solo_tackle_1_player_name', 'solo_tackle_2_player_name', 'assist_tackle_1_player_id',
        'assist_tackle_1_player_name', 'assist_tackle_1_team', 'assist_tackle_2_player_id',
        'assist_tackle_2_player_name', 'assist_tackle_2_team', 'assist_tackle_3_player_id', 'assist_tackle_3_player_name',
        'assist_tackle_3_team', 'assist_tackle_4_player_id', 'assist_tackle_4_player_name', 
        'assist_tackle_4_team', 'pass_defense_1_player_id', 'pass_defense_1_player_name', 'pass_defense_2_player_id',
        'pass_defense_2_player_name', 'fumbled_1_team', 'fumbled_1_player_name', 'fumbled_2_player_id', 'fumbled_2_player_name',
        'fumbled_2_team', 'fumble_recovery_1_team', 'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name',
        'fumble_recovery_2_team', 'fumble_recovery_2_yards', 'fumble_recovery_2_player_id',
        'fumble_recovery_2_player_name', 'return_team', 'penalty_player_id', 'penalty_player_name', 
        'replay_or_challenge', 'replay_or_challenge_result', 'penalty_type', 'defensive_two_point_attempt',
        'defensive_two_point_conv', 'defensive_extra_point_attempt', 'defensive_extra_point_conv','Unnamed: 0.1', 
        'Away', 'Home', 'Team_x', 'Team_y'], axis=1)
        
    
    data.to_csv('removed_20'+num+'.csv')