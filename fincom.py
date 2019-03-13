import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#d2018 = pd.read_csv("almost_2018.csv", low_memory=False)
d2017 = pd.read_csv("almost_2017.csv", low_memory=False)
d2016 = pd.read_csv("almost_2016.csv", low_memory=False)
d2015 = pd.read_csv("almost_2015.csv", low_memory=False)
d2014 = pd.read_csv("almost_2014.csv", low_memory=False)
d2013 = pd.read_csv("almost_2013.csv", low_memory=False)
d2012 = pd.read_csv("almost_2012.csv", low_memory=False)


almost = pd.concat([d2017, d2016, d2015, d2014, d2013, d2012], axis=0, ignore_index=True,sort=False)

almost = almost.drop(columns=['ep', 'epa', 'total_home_epa', 'total_away_epa', 'total_home_rush_epa',
    'total_away_rush_epa', 'total_home_pass_epa', 'total_away_pass_epa', 'air_epa', 'yac_epa',
    'comp_air_epa', 'comp_yac_epa',	'total_home_comp_air_epa', 'total_away_comp_air_epa', 
    'total_home_comp_yac_epa', 'total_away_comp_yac_epa', 'total_home_raw_air_epa', 
    'total_away_raw_air_epa', 'total_home_raw_yac_epa', 'total_away_raw_yac_epa', 'wp', 'def_wp',
	'home_wp', 'away_wp', 'wpa', 'home_wp_post', 'away_wp_post', 'total_home_rush_wpa',
    'total_away_rush_wpa', 'total_home_pass_wpa', 'total_away_pass_wpa', 'air_wpa', 'yac_wpa',
	'comp_air_wpa', 'comp_yac_wpa', 'total_home_comp_air_wpa', 'total_away_comp_air_wpa', 
    'total_home_comp_yac_wpa', 'total_away_comp_yac_wpa', 'total_home_raw_air_wpa', 
    'total_away_raw_air_wpa', 'total_home_raw_yac_wpa',	'total_away_raw_yac_wpa', 'play_id',
    'game_id', 'game_date', 'desc', 'no_score_prob', 'opp_fg_prob', 'opp_safety_prob', 'opp_td_prob',
	'fg_prob', 'safety_prob', 'td_prob'])

almost['duration'] = almost['game_seconds_remaining'].diff()
almost.loc[almost.game_seconds_remaining == 3600, 'duration'] = 0
almost.to_csv('final.csv')
