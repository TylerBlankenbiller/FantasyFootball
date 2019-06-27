transfer = gameLog['pass_attempt', 'complete_pass', 'fumble', 'fumble_lost',
                                    'qb_hit', 'interception', 'extra_point_attempt', 
                                    'extra_point_result', 'rush_attempt', 'complete_pass',
                                    'pass_touchdown', 'yards_gained', 'rush_touchdown',
                                    'sack', 'pass_attempt', 'field_goal_attempt', 'field_goal_result',
                                    'kick_distance']
            transfer['Year'] = year
            transfer['game_id'] = game_id
                
            transfer['posteam'] = posteam
            transfer['defteam'] = defteam
            for i in range(len(throw)):
                if gameLog['Pass2' + i.astype(str)] == 1:
                    transfer['passer_player_id'] = i
                    break
            for i in range(len(rusher)):
                if gameLog['Rush' + i.astype(str)] == 1:
                    transfer['rusher_player_id'] = i
                    break
            for i in range(len(receiver)):
                if gameLog['Rec' + i.astype(str)] == 1:
                    transfer['receiver_player_id'] = i
                    break
            for i in range(len(kicker)):
                if gameLog['Kick' + i.astype(str)] == 1:
                    transfer['kicker_player_id'] = i
                    break
            try:
                saveGame = pd.concat([saveGame, transfer], axis=0, ignore_index=True)
            except NameError:#saveGame doesn't exist
                saveGame = transfer