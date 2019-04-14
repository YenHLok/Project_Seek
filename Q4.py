# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:17:26 2019

@author: Lok Hsiao Yen
"""

#####################################################################################################
## Part (a)

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None) # to ensure console display all columns

import os
import sys
sys.path.append(os.path.abspath('..'))


import pickle
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



WIKI_URL = "https://en.wikipedia.org/wiki/List_of_best-selling_music_artists"
req = requests.get(WIKI_URL)
soup = BeautifulSoup(req.content, 'lxml')
table_classes = {"class": ["sortable", "plainrowheaders"]}
wikitables = soup.findAll("table", table_classes)

out = []
for index, table in enumerate(wikitables):  
    for row in table.findAll("tr"):
        cells = row.findAll(["th"])
        for cell in cells:
            links = cell.findAll('a') 
       
            for link in links:
                title = link.get('title')
                if not title is None: 
                    title = re.sub("&amp;","&",title).strip()
                    title = re.sub("\([^)]+\)","",title).strip() # remove brakets
                    content = re.search(">([\w --&;\/\'\.]+)</a>", str(link), re.IGNORECASE).group(1) # the content of link            
                    content = re.sub("&amp;","&",content).strip()
                    if title==content:
                        out.append(title)
    
df = pd.DataFrame()
df['Artist'] = out


#####################################################################################################
## Part (b)

import requests
import mwparserfromhell
response = requests.get(
    'https://en.wikipedia.org/w/api.php',
    params={
        'action': 'query',
        'format': 'json',
        'titles': '1990s_in_music',
        'prop': 'revisions',
        'rvprop': 'content',
     }
).json()

page = next(iter(response['query']['pages'].values()))
wikicode = page['revisions'][0]['*']
parsed_wikicode = mwparserfromhell.parse(wikicode)
raw_text = parsed_wikicode.strip_code().strip()


#####################################################################################################
## Part (c)

for i in range(len(df)):
    artist = df.loc[i].Artist
    pattern = ''.join(['(']+[x + '.?' for x in artist[:-1]]+[artist[-1]]+[')'])

    out = re.search(pattern, raw_text)
    if out:
        df.loc[i,'match'] = 1          
        df.loc[i,'match_text'] = out.group(1).strip()             
    else:
        df.loc[i,'match'] = 0 
        df.loc[i,'match_text'] = ''        


save_obj(df, 'Q4_Data')
print(f"{int(df['match'].sum())} out of the {len(df)} best-selling music artists are from the 1990s")


























    
    