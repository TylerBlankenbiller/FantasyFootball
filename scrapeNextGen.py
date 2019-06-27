import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(columns=['Home', 'Away', 'Weather', 'Wind', 'Year', 'Week'])
#df['away'] = df['away'].astype('str')
#df['home'] = df['home'].astype('str')
#df['weather'] = df['weather'].astype('str')
#df['wind'] = df['wind'].astype('str')

num = 'test'
for idx in range(9):#years 2012-2018
    if idx < 6:
        idx = 6
    num = str(idx + 10)
    
    print("IDX ", num)
    for week in range(17):#17 weeks
        week += 1
        #URL
        quote_page = 'https://nextgenstats.nfl.com/stats/passing/20'+str(num)+'/'+str(week)+'/#yards'
        print(quote_page)
        
        page = requests.get(quote_page)
        time.sleep(2)
        
        soup = BeautifulSoup(page.content, 'html.parser')
       
        test = soup.find('tbody')
        
        print(test)
        
        #print(test)
        test2 = test.find_all("div")#, {"class": "cell"})
        max = len(test2)
        
        d = {}
        
        i = 8
        
        print(test2[10].get_text())