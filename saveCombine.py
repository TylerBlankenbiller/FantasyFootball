#Add the combine to the csv with everything else

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

combine = pd.read_csv("combine20.csv", low_memory = False)

for idx in range(7):
    idx+=2
    num = str(idx + 10)
        
    data = pd.read_csv("reg_coaches_20"+num+".csv", low_memory=False)
    
    combine['FName'], combine['LName'] = combine['Player'].str.split(' ', 1).str
    
    combine.to_csv('combine201.csv')
    