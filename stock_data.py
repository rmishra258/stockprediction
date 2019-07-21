# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:10:32 2019

@author: Rahul4.Mishra
"""

import pandas as pd
import numpy as np
import requests as r
import matplotlib.pyplot as plt
import sentiment_analyser
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import operator
import warnings
from bs4 import BeautifulSoup


bse_data = pd.read_csv(r'equity.csv')



def make_df(df, sentiment_analyser):

    cols = ['date','open', 'high', 'low', 'close', 'adj close','volume']  
    
    
    
    
    
    df.columns = cols
    
     
    df['open'] = [int(x.replace('.','')[0]) for x in df['open']]
    df['close'] = [int(x.replace('.','')[0]) for x in df['close']]
    df['high'] = [int(x.replace('.','')[0]) for x in df['high']]
    df['low'] = [int(x.replace('.','')[0]) for x in df['low']]
    df['adj close'] = [int(x.replace('.','')[0]) for x in df['adj close']]
    df['volume'] = [int(x.replace('.','')[0]) for x in df['volume']]
    
    
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = [x.day for x in df['date']]
    df['month'] = [x.month for x in df['date']]
    df['year'] = [x.year for x in df['date']]
    
    #delete the first rows after shifting yestrdays value
    df['yesterday'] = df['close'].shift(1)
    df.drop(df.index[0], inplace=True)
    

    #round the close values
    df['yesterday'] = [round(x) for x in df['yesterday']]
    df['close'] = [round(x) for x in df['close']]
    df['open'] = [round(x) for x in df['open']]
    
    
    df['more'] = np.where(df['close'] > df['open'], 1, 0)
    
    #add financial quarter

    df['quarter'] = pd.cut(df['month'], bins=[1,3,6,9,12], labels=[1,2,3,4])
    df.quarter.value_counts()
    
    #also add day of the week
    df['weekday'] = pd.to_datetime(df.index)
    df['weekday'] = df.weekday.dt.dayofweek

    #devide day of month by 3

    df['day_quartile'] = pd.cut(df['day'], bins=[0,10,20,31], labels=[-1,0,1])
    df.index = np.arange(0, df.shape[0])
    df['pos'] = sentiment_analyser.pos_ratio
    df['neg'] = sentiment_analyser.neg_ratio
    
    df['day_quartile'] = df['day_quartile'].astype(int)
    df['quarter'] = df['quarter'].astype(int)
    
    

    df_main = df[['day_quartile','month','quarter','weekday','pos','neg','more']]
    
    return df, df_main






'''
plt.plot(df['close'])
plt.xlabel('90 days')
plt.ylabel('Share price')
plt.title('Price movement of ' + symbol)
plt.show()


df['more'].value_counts()

print(df.tail())
print(df_main.head())

'''

def make_model(symbol, df_main) : 
    
    x = df_main[['day_quartile','month','quarter','weekday','pos','neg']]
    y = df_main['more']
    
    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size = .3)
    
    clf = XGBClassifier()
    clf = clf.fit(x_train, y_train)
    
    #get feature importances
    imp_features = {x:y for x,y in zip(x.columns, clf.feature_importances_)}
    imp_features_ = sorted(imp_features.items(), key=operator.itemgetter(1))[-3:]
    imp_features_ = [x[0] for x in imp_features_]
    
    #reinitialize with important features only
    x = df_main[imp_features_]
    y = df_main['more']
    
    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size = .3)
    
    clf = clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    
    warnings.simplefilter('ignore')
    
    acc = accuracy_score(y_test, pred)
    print("Accuracy score ", acc)

    
    #make dataframe
    
    listing = symbol.split('.')[0]
    print('this is listing', listing)
    industry = bse_data[bse_data['Security Id'].isin([listing])]['Industry'][0]
    company = bse_data[bse_data['Security Id'].isin([listing])]['Security Name'][0]
    
    df_by_acc = pd.DataFrame({'listing' : [listing], 'industry' : [industry], 'company' : [company] ,'accuracy_score' : [acc]})

    
    return df_by_acc
    
    

def get_data(url) : 
    dat = r.get(url=url).text
    
    soup = BeautifulSoup(dat, 'lxml')
    
    soup = [x.find_all('span') for x in soup.find_all('table')][0]
    soup = soup[7:-3]
    soup = [x.text for x in soup]
    soup = np.array(soup)
    soup = soup.reshape(soup.shape[0],7)
    
    df = pd.DataFrame(soup)

    return df


def run_all():
    
    url1 = 'https://in.finance.yahoo.com/quote/' 
    url2 = '/history?p='
    df_by_acc_main = pd.DataFrame({'listing' : [], 'industry' : [], 'company' : [] ,'accuracy_score' : []})

    error_listing = []
    for names in bse_data['Security Id']:
        
        name = names+ '.BO'
        
        final_url = url1 + name + url2+ name
        
        print('this is final url', final_url)
        df = get_data(final_url)
        df, df_main = make_df(df, sentiment_analyser)
        
        try : 
            df_by_acc = make_model(name, df_main)
            df_by_acc_main = df_by_acc_main.append(df_by_acc)
            
            print(df_by_acc_main)
            
        except : 
            
            print('error for ', name)
            error_listing.append(name)
        
        

run_all()

"""
#test case
for names in bse_data['Security Id']:
    
    df, df_main = make_df(api_key, fn, url, payload,sentiment_analyser)
    print(names)
    

    
def get_data_from_server(payload):
        error_listings = []
        
        api_key, fn, url, payload = init(names)
        d = r.get(url, params=payload).json()
        
        
        try :
            df = pd.DataFrame.from_dict(d['Time Series (Daily)'], orient='index')
            df, df_main = make_df(api_key, fn, url, payload, sentiment_analyser)
            
        except :
            continue
"""