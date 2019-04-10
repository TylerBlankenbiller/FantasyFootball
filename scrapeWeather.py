#Credit:
#https://medium.freecodecamp.org/how-to-scrape-websites-with-python-and-beautifulsoup-5946935d93fe
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
    if idx < 2:
        idx = 2
    num = str(idx + 10)
    
    print("IDX ", num)
    for week in range(17):#17 weeks
        week += 1
        #URL
        quote_page = 'http://www.nflweather.com/en/week/20'+str(num)+'/week-'+str(week)+'/'
        print(quote_page)
        
        page = requests.get(quote_page)
        
        soup = BeautifulSoup(page.content, 'html.parser')
       
        test = soup.find('tbody')
        
        test2 = test.find_all(class_='text-center')
        max = len(test2)
        
        d = {}
        
        i = 8
        
        while i <= max:#games per week
            
            away = test2[i-8].get_text()
            home = test2[i-7].get_text()
            weather = test2[i-3].get_text()
            wind = test2[i-2].get_text()
            
            away = away.replace('\n','')
            away = away.replace(' ','')
            
            home = home.replace('\n','')
            home = home.replace(' ','')
            
            weather = weather.replace('\n', '')
            weather = weather.strip()
            
            series = pd.DataFrame({'Home':[home], 'Away':[away], 'Weather':weather, 'Wind':wind, 'Year':'20'+num, 'Week':week, 'SType':'Regular'})
            df = df.append(series, ignore_index=True)
            print(df)
            i += 8

        time.sleep(2)
        
        if week < 4:#3 preseason weeks(Week 4 most starters don't play, so ignore)
            quote_page = 'http://www.nflweather.com/en/week/20'+str(num)+'/pre-season-week-'+str(week)+'/'
            print(quote_page)
            
            page = requests.get(quote_page)
            
            soup = BeautifulSoup(page.content, 'html.parser')
           
            test = soup.find('tbody')
            
            test2 = test.find_all(class_='text-center')
            max = len(test2)
            
            d = {}
            
            i = 8
            
            while i <= max:#games per week
                
                away = test2[i-8].get_text()
                home = test2[i-7].get_text()
                weather = test2[i-3].get_text()
                wind = test2[i-2].get_text()
                
                away = away.replace('\n','')
                away = away.replace(' ','')
                
                home = home.replace('\n','')
                home = home.replace(' ','')
                
                weather = weather.replace('\n', '')
                weather = weather.strip()
                
                series = pd.DataFrame({'Home':[home], 'Away':[away], 'Weather':weather, 'Wind':wind, 'Year':'20'+num, 'Week':week, 'SType':'Pre'})
                df = df.append(series, ignore_index=True)
                print(df)
                i += 8

            time.sleep(2)
        

df.to_csv('weather.csv')
df = pd.read_csv('weather.csv', low_memory=False)
out = df.to_json(orient='records')[1:-1].replace('},{', '} {')
with open('weather.txt', 'w') as f:
    f.write(out)
        