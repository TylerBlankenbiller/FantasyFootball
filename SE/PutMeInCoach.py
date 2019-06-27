#Credit:
#https://medium.freecodecamp.org/how-to-scrape-websites-with-python-and-beautifulsoup-5946935d93fe
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

num = 'test'
save = pd.DataFrame()
teams = ['phi', 'dal', 'nyg', 'was', 'nor', 'atl', 'car', 'tam', 'chi', 'min', 'det', 'gnb', 'ram', 'sfo', 'sea', 'crd',
        'nwe', 'mia', 'nyj', 'buf', 'pit', 'cin', 'cle', 'rav', 'den', 'kan', 'rai', 'sdg', 'oti', 'jax', 'clt', 'htx']

for t in teams:
    quote_page = 'https://www.pro-football-reference.com/teams/'+t+'/coaches.htm'
    print(quote_page)
    
    page = requests.get(quote_page)
        
    soup = BeautifulSoup(page.content, 'html.parser')
     
    for body in soup("tbody"):
        body.unwrap()

    df = pd.read_html(str(soup), flavor="bs4")
    df[1]['Team'] = t
    save = save.append(df[1], ignore_index=True)
    print(save)
    
    #test = soup.find('tbody')
        
    #test2 = test.find_all(class_='left')
    
    #games = test.find_all('td', {'data-stat':"g"})
    #max = len(test2)
    #print(games)
    d = {}
    time.sleep(1)
save.to_csv('coaches.csv')