# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:35:33 2019

@author: Lok Hsiao Yen
"""


#####################################################################################################
## Set-up

import numpy as np
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

df = pd.read_csv('data_job_posts.csv', sep=',', na_filter=False)



#####################################################################################################
## Part (b)

stop_list = ['Job Title','Reports To','-----','<EOD>']
stop_pattern = ''.join([f"{tag}|" for tag in stop_list[:-1]]+[stop_list[-1]])

include_list = ['EU','UNFCCC','CSW']
include_pattern = ''.join([f"{tag}|" for tag in include_list[:-1]]+[include_list[-1]])

dfOut = pd.DataFrame()
startId=0


def Remove_Numbering(s):
    prev_s = s
    new_s = re.sub(r'[\d+](?:\.|\)|\.\))([^\d]+.*)',r'\1',prev_s).strip()           
    
    while True:
        if prev_s==new_s:
            break        
        prev_s = new_s        
        new_s = re.sub(r'[\d+](?:\.|\)|\.\))([^\d]+.*)',r'\1',prev_s).strip()   
    return new_s

#s=out
def Clean_Text(s):
    s = Remove_Numbering(s)  
    s = re.sub('\t',' ',s)   
    s = re.sub('[ ]{2,}',' ',s)      
    return s


def Fix_Spacing(s):
    s = re.sub(r"([^A-Z\(]*)([(A-Z]{3,})",r'\1 \2',s)    
    return s


def Extract_Date(s):
    s = '<SOS>'+s+'<EOS>'        
    year = re.search('(?:[12][09][0189][0-9]|[89][0-9])', s)
    if not year is None:          
        year = year.group(0).strip()   
    else:
        year = ''

    month = re.search('(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[/w]*', s, re.IGNORECASE)
    if not month is None:          
        month = month.group(0).strip()   
    else:
        month = ''    
    
    day = re.search('[^0-9]*([0123]?[0-9])[^0-9]*', s)
    if not day is None:          
        day = day.group(1).strip()   
    else:
        day = ''    
       
    out = day+' '+month+' '+year
    if len(out)<5: 
        out = 'NA'
    return out.strip()  


#i=1025
for i in range(startId,len(df)):
    if i%100==0:
        print(f'{i} in {len(df)-1}')
    row = df.loc[i]
    jobpost = row.jobpost   
    jobpost = re.sub('\r\n',' ',jobpost) 
    jobpost = Fix_Spacing(jobpost)    
    jobpost = re.sub('[ ]{2,}',' ',jobpost)     
    jobpost = jobpost+' <EOD>---'
    
    def Get_Query(query):
        pattern = f"{query}:(.*?) ?({include_pattern})? (?:[A-Z \/]+|{stop_pattern})? ?(?:and)?(\/)?(?:or)?(?:[A-Z \/]+|{stop_pattern})(?::|---)"  
        res = re.search(pattern, jobpost)
        if not res is None:          
            out = ' '.join([x for x in res.groups() if x is not None])      
            out = Clean_Text(out)
        else:
            out = 'NA'
        return out
    
    dfOut.loc[i,'Job Title'] = Get_Query('TITLE')  
    dfOut.loc[i,'Position Duration'] = Get_Query('DURATION')   
    dfOut.loc[i,'Position Location'] = Get_Query('LOCATION') 
    dfOut.loc[i,'Job Description'] = Get_Query('DESCRIPTION') 
    dfOut.loc[i,'Job Responsibilities'] = Get_Query('RESPONSIBILITIES')     
    dfOut.loc[i,'Required Qualifications'] = Get_Query('(?:ELIGIBILITY CRITERIA|ELIGIBILITY|QUALIFICATIONS)') 
    dfOut.loc[i,'Remuneration'] = Get_Query('REMUNERATION')     
    dfOut.loc[i,'Application Deadline'] = Extract_Date(Get_Query('DEADLINE'))    
    dfOut.loc[i,'About Company'] = Get_Query('COMPANY')     

save_obj(dfOut, 'Q2_Data')    


#####################################################################################################
## Part (c)

dfOut = load_obj('Q2_Data')    
dfOut['Company'] = df['Company']
dfOut['Date'] = df['date']

def Extract_Month(s):
    out = re.search('(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[/w]*', s, re.IGNORECASE)
    if not out is None:          
        out = out.group(0).strip()   
    else:
        out = 'NA'    
    return out.strip()  


def Extract_Year(s):
    out = re.search('(?:[12][09][0189][0-9]|[89][0-9])', s)
    if not out is None:          
        out = out.group(0).strip()   
    else:
        out = 'NA'    
    return out.strip()  

dfOut['Date_Month'] = dfOut['Date'].apply(lambda s: Extract_Month(s))
dfOut['Date_Year'] = dfOut['Date'].apply(lambda s: Extract_Year(s))

top_company = pd.DataFrame(dfOut.loc[(dfOut['Date_Year']>'2011')&(dfOut['Date_Year']!='NA')] \
    .groupby('Company')['Company'].count()) \
    .rename(columns={'Company': 'Count'}) \
    .reset_index().sort_values('Count',ascending=False) \
    .head(1).Company.iloc[0]

print(f'The company with the most number of job ads in the past 2 years is {top_company}')


#####################################################################################################
## Part (d)


top_month = pd.DataFrame(dfOut.loc[dfOut['Date_Month']!='NA'] \
    .groupby('Date_Month')['Date_Month'].count()) \
    .rename(columns={'Date_Month': 'Count'}) \
    .reset_index().sort_values('Count',ascending=False) \
    .head(1).Date_Month.iloc[0]

print(f'The month with the largest number of job ads over the years is {top_month}')


#####################################################################################################
## Part (e)


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(s):
    s = re.sub('[-:;,#\(\)]',' ',s) # remove special symbols
    s = re.sub("[^a-zA-Z]", " ",s)  # only letters
    s = re.sub("NA", " ",s)  # remove NA                
    s = re.sub('[ ]{2,}',' ',s).strip()
    tokens = nltk.word_tokenize(s)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))  
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    out = ' '.join(lemmas)
    return out

dfOut['Job_Responsibilities_Normalized'] = dfOut['Job Responsibilities'].apply(lambda s: normalizer(s))
dfOut['Job_Description_Normalized'] = dfOut['Job Description'].apply(lambda s: normalizer(s))



#####################################################################################################
## Part (f)

def convert_na(s):
    if s=='NA':
        out = 'No Info'
    else:
        out = s
    return out


dfOut['Position Duration'] = dfOut['Position Duration'].apply(lambda s: convert_na(s))


#####################################################################################################
## Part (g)

save_obj(dfOut, 'Q2_Data')    













