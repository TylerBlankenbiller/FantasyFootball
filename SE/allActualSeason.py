import pandas as pd

def com(x):
    d = {}
    d['name'] = x['name'].unique()
    d['date'] = x['date'].unique()
    d['position'] = x['position'].unique()
    d['game_id'] = x['game_id'].unique()
    d['posteam'] = x['posteam'].unique()
    d['name'] = d['name'][0]
    d['date'] = d['date'][0]
    d['position'] = d['position'][0]
    d['game_id'] = d['game_id'][0]
    d['posteam'] = d['posteam'][0]
    d['ATT'] = x['ATT'].sum()
    d['COMP'] = x['COMP'].sum()
    d['FUM'] = x['FUM'].sum()
    d['HIT'] = x['HIT'].sum()
    d['INT'] = x['INT'].sum()
    d['LONG'] = x['LONG'].sum()
    d['LST'] = x['LST'].sum()  
    d['PAT_Attempt'] = x['PAT_Attempt'].sum()  
    d['PAT_Made'] = x['PAT_Made'].sum()  
    d['PAT_percent'] = x['PAT_percent'].sum()  
    d['RATT'] = x['RATT'].sum()  
    d['REC'] = x['REC'].sum()  
    d['RECTD'] = x['RECTD'].sum()  
    d['RECYDS'] = x['RECYDS'].sum()  
    d['RTD'] = x['RTD'].sum()  
    d['RYDS'] = x['RYDS'].sum()  
    d['SACK'] = x['SACK'].sum()  
    d['TD'] = x['TD'].sum()  
    d['TGTS'] = x['TGTS'].sum()  
    d['YDS'] = x['YDS'].sum()  
    d['attempt'] = x['attempt'].sum()  
    d['long'] = x['long'].sum()  
    d['longest'] = x['longest'].sum()  
    d['med'] = x['med'].sum()  
    d['percent'] = x['percent'].sum()  
    d['short'] = x['short'].sum()  
    return pd.Series(d, index=['name', 'date', 'position', 'game_id', 'posteam', 'ATT', 'COMP', 'FUM', 'HIT', 'INT', 'LONG', 'LST',
                                'PAT_Attempt', 'PAT_Made', 'PAT_percent', 'RATT','REC', 'RECTD', 'RECYDS', 'RTD', 'RYDS', 'SACK',
                                'TD', 'TGTS', 'YDS', 'attempt', 'long', 'longest', 'med', 'percent', 'short'])

df = pd.read_csv('allActualStats.csv', low_memory=False)
df2 = df[['id', 'year', 'name', 'posteam', 'position']].copy()

test1 = df.groupby(['id', 'year']).apply(com)

test1.to_csv('allActualSeason.csv')
