# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:54:06 2020

@author: Chens
"""

import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re
import requests
from bs4 import BeautifulSoup
import os

def get_data():
    
    headers={
    	'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    
    url = 'https://3g.dxy.cn/newh5/view/pneumonia?scene=2&clicktime=1579582238&enterid=1579582238&from=singlemessage&isappinstalled=0' 
    
    html = requests.get(url, headers = headers)
    soup = BeautifulSoup(html.text, 'lxml')
    data_info = soup.find_all('span', {'class':'content___2hIPS'})
    
    pattern = re.compile('<span style.*?>(\d*)')
    data = re.findall(pattern, str(data_info))

    data_lst = []    
    for i in data:
        i = int(i)
        data_lst.append(i)
    
    return data_lst 



def clean_data(dlst):
    df = pd.DataFrame(dlst)
    df.columns = ['infected', 'suspected', 'healed', 'death']
    df['times'] = time.strftime('%Y-%m-%d %H:%M')

    return  df




if __name__ == '__main__':
    curtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))    
    os.chdir(r'C:\Users\Chens\Desktop\python学习\python-project')
    print(curtime)
    wuhan = pd.read_csv('wuhan.csv')
    dlst = []
    dlst.append(get_data())
    wuhan = wuhan.append(clean_data(dlst), ignore_index = True, sort = False)
    wuhan = wuhan[['infected', 'suspected', 'healed', 'death', 'times']]
    wuhan = wuhan.drop_duplicates(subset = ['infected'])
    wuhan.sort_index(axis=0, ascending = True, by = 'times')
    wuhan.to_csv('wuhan.csv')
    
    df = pd.read_csv('wuhan.csv', encoding='utf-8', index_col='times')[['infected']]
    df.index = pd.to_datetime(df.index)
        
