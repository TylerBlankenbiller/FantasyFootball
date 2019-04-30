import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from firebase import firebase

class Sugar:

    data = pd.read_csv('qwerty.csv', low_memory=False)
    firebase_app = firebase.FirebaseApplication('https://real-stat-rats.firebaseapp.firebaseio.com/',authentication=None)
    firebase = firebase.FirebaseApplication('https://real-stat-rats.firebaseio.com', None)
        
    postdata = data.to_dict()
    
    result = firebase_app.post('/', {'name':{'nameChild':'tim'}})
    #resultPut = firebase.put('player', 'ID', data['ID'][0])
    print(result)
    #print(resultPut)