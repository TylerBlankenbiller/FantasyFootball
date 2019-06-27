import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

found = False
coach = pd.read_csv("coaches2.csv", low_memory = False)

for idx in range(7):
    idx+=2
    num = str(idx + 10)
        
    data = pd.read_csv("almost_20"+num+".csv", low_memory=False, index_col=0)

    print("year")
    print(num)
    data=data.fillna(0)
    data = data.loc[data['HCoach'] != 0]
    data = data.drop(['play_id', 'game_date', 'game_half', 'time', 'yrdln', 'pass_length',
        'lateral_receiver_player_id', 'lateral_receiver_player_name', 'lateral_rusher_player_id', 'lateral_rusher_player_name',
        'lateral_sack_player_id', 'lateral_sack_player_name', 'lateral_interception_player_id', 'lateral_interception_player_name',
        'lateral_punt_returner_player_id', 'lateral_punt_returner_player_name',
        'lateral_kickoff_returner_player_id', 'lateral_kickoff_returner_player_name', 
        'own_kickoff_recovery_player_id', 'own_kickoff_recovery_player_name',
        'tackle_for_loss_2_player_id', 'tackle_for_loss_2_player_name',
        'qb_hit_2_player_id', 'qb_hit_2_player_name', 'forced_fumble_player_1_team', 
        'forced_fumble_player_2_team', 'forced_fumble_player_2_player_id',
        'forced_fumble_player_2_player_name', 'solo_tackle_1_team', 'solo_tackle_2_team',
        'solo_tackle_2_player_id', 'solo_tackle_2_player_name',
        'assist_tackle_1_team', 'assist_tackle_2_player_id',
        'assist_tackle_2_player_name', 'assist_tackle_2_team', 'assist_tackle_3_player_id', 'assist_tackle_3_player_name',
        'assist_tackle_3_team', 'assist_tackle_4_player_id', 'assist_tackle_4_player_name', 
        'assist_tackle_4_team', 'pass_defense_2_player_id',
        'pass_defense_2_player_name', 'fumbled_1_team', 'fumbled_2_player_id', 'fumbled_2_player_name',
        'fumbled_2_team', 'fumble_recovery_1_team',
        'fumble_recovery_2_team', 'fumble_recovery_2_yards', 'fumble_recovery_2_player_id',
        'fumble_recovery_2_player_name', 'return_team',
        'replay_or_challenge_result', 'penalty_type', 'defensive_two_point_attempt',
        'defensive_two_point_conv', 'defensive_extra_point_attempt', 'defensive_extra_point_conv', 
        'Away', 'Home', 'Team_x', 'Team_y', 'play_id', 'time', 'yrdln', 'desc', 'play_type', 'home_team',
        'away_team', 'side_of_field', 'game_date', 'game_half', 'quarter_end', 'sp', 'goal_to_go',
        'ydsnet', 'shotgun', 'no_huddle', 'qb_dropback', 'qb_scramble',
        'pass_length', 'td_team', 'posteam_score_post', 'defteam_score_post', 'score_differential_post', 
        'no_score_prob', 'opp_fg_prob', 'opp_safety_prob', 'opp_td_prob',
        'fg_prob', 'safety_prob', 'td_prob', 'extra_point_prob',
        'two_point_conversion_prob', 'ep', 'epa', 'total_home_epa',
        'total_away_epa', 'total_home_rush_epa', 'total_away_rush_epa',
        'total_home_pass_epa', 'total_away_pass_epa', 'air_epa', 'yac_epa',
        'comp_air_epa', 'comp_yac_epa', 'total_home_comp_air_epa', 
        'total_away_comp_air_epa', 'total_home_comp_yac_epa',
        'total_away_comp_yac_epa', 'total_home_raw_air_epa', 
        'total_away_raw_air_epa', 'total_home_raw_yac_epa',
        'total_away_raw_yac_epa', 'wp', 'def_wp', 'home_wp', 
        'away_wp', 'wpa', 'home_wp_post', 'away_wp_post',
        'total_home_rush_wpa', 'total_away_rush_wpa',
        'total_home_pass_wpa', 'total_away_pass_wpa',
        'air_wpa', 'yac_wpa', 'comp_air_wpa', 'comp_yac_wpa',
        'total_home_comp_air_wpa', 'total_away_comp_air_wpa',
        'total_home_comp_yac_wpa', 'total_away_comp_yac_wpa',
        'total_home_raw_air_wpa', 'total_away_raw_air_wpa',
        'total_home_raw_yac_wpa', 'total_away_raw_yac_wpa','first_down_penalty',
        'punt_inside_twenty', 'punt_in_endzone',
        'punt_out_of_bounds', 'punt_downed', 'punt_fair_catch', 
        'kickoff_inside_twenty', 'kickoff_in_endzone', 'kickoff_out_of_bounds',
        'kickoff_downed', 'kickoff_fair_catch', 'fumble_forced',
        'fumble_not_forced', 'fumble_out_of_bounds', 'solo_tackle',
        'tackled_for_loss', 'own_kickoff_recovery',
        'own_kickoff_recovery_td', 'touchdown', 'return_touchdown',
        'assist_tackle', 'lateral_reception',
        'lateral_rush', 'lateral_return', 'lateral_recovery',
        'lateral_receiver_player_id', 'lateral_receiver_player_name',
        'lateral_rusher_player_id', 'lateral_rusher_player_name',
        'lateral_sack_player_id', 'lateral_sack_player_name',
        'lateral_interception_player_id', 'lateral_interception_player_name',
        'lateral_punt_returner_player_id', 'lateral_punt_returner_player_name',
        'lateral_kickoff_returner_player_id', 'lateral_kickoff_returner_player_name',
        'own_kickoff_recovery_player_id', 'own_kickoff_recovery_player_name',
        'blocked_player_id', 'blocked_player_name', 'tackle_for_loss_2_player_id',
        'tackle_for_loss_2_player_name', 'qb_hit_2_player_id',
        'qb_hit_2_player_name', 'forced_fumble_player_1_team',
        'forced_fumble_player_2_team', 'forced_fumble_player_2_player_id',
        'forced_fumble_player_2_player_name', 'solo_tackle_1_team',
        'solo_tackle_2_team', 'solo_tackle_2_player_id', 
        'solo_tackle_2_player_name', 'assist_tackle_1_team',
        'assist_tackle_2_player_id', 'assist_tackle_2_player_name',
        'assist_tackle_2_team', 'assist_tackle_3_player_id',
        'assist_tackle_3_player_name', 'assist_tackle_3_team',
        'assist_tackle_4_player_id', 'assist_tackle_4_player_name',
        'assist_tackle_4_team', 'pass_defense_2_player_id',
        'pass_defense_2_player_name', 'fumbled_1_team',
        'fumbled_2_player_id', 'fumbled_2_player_name', 'fumbled_2_team',
        'fumble_recovery_2_team', 'fumble_recovery_2_yards', 
        'fumble_recovery_2_player_id', 'fumble_recovery_2_player_name',
        'return_team', 'replay_or_challenge_result', 'penalty_type', 
        'defensive_two_point_attempt', 'defensive_two_point_conv', 
        'defensive_extra_point_attempt', 'defensive_extra_point_conv',
        'Away', 'Home', 'Team_x', 'Team_y'], axis=1)
        
    
    data.to_csv('removed_20'+num+'.csv')