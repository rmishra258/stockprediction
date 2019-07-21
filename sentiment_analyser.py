# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:14:27 2019

@author: Rahul4.Mishra
"""

consumer_key = 'c57SU7sulViKSmjsOTi4kTO3W'
consumer_secret = 'cNT3yk5ibQ315AWNCJHgE9ipCGlM1XnenHZu9cBWaVL3q7fPew'
access_token = '796747210159517701-DhOBQgwzeb6q4eXlI4WjwPRJH1CuEIT'
access_token_secret = 'sMrnPZ4ExI8um43wquUvFEUCTyY61HYRf7z3jv00ltXlt'

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize
import numpy as np
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

term= 'BAJAJ FINANCE'

def run_all(term): 
    
    api = auth(consumer_key, consumer_secret, access_token, access_token_secret)
    print("API handshake successful")
    print("Searching for term ", term)
    tweet_data = tweet(term)
    print("Removing stopwords")
    sw = stopwords.words('english')
    tweets = remove_sw(tweet_data, sw)
    print("Analysing sentiment for ", term)
    sentiment, neg, neu, pos, comp = sentiment_analyser(tweets)
    df = build_df(pos,neg,neu,comp, tweets)
    print('Done \n')
    
    return df
                  

#authentication

def auth(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token,access_token_secret)
    api = tweepy.API(auth)
    
    return api


def tweet(term):

    tweets = api.search(term, count=1000)

    #lowercase
    tweet_data = [sent_tokenize(x.text.lower()) for x in tweets]

    return tweet_data

#remove stopwords


#print(tweet_data[0])
api = auth(consumer_key, consumer_secret, access_token, access_token_secret)

def remove_sw(lst, corpus):
    
    data = []
    data_punc = []
    
    for x in lst:
        
        data.append([y for y in x if x not in corpus])
        
    data = [" ".join(x) for x in data]
        
    regtkn = RegexpTokenizer(r'\w+')
    
    data = [" ".join(regtkn.tokenize(x)) for x in data]
    
    return data



def sentiment_analyser(lst):
    
    sid = SentimentIntensityAnalyzer()
    sentiment = [sid.polarity_scores(x) for x in lst]
    neg = [sid.polarity_scores(x)['neg'] for x in lst]
    neu = [sid.polarity_scores(x)['neu'] for x in lst]
    pos = [sid.polarity_scores(x)['pos'] for x in lst]
    comp = [sid.polarity_scores(x)['compound'] for x in lst]
    
    return sentiment, neg, neu, pos, comp


def build_df(pos,neg,neu,comp, tweet):
    
    df = pd.DataFrame(data=tweet, columns=['tweet'])
    df['pos'] = pos
    df['neg'] = neg
    df['neu'] = neu
    df['comp'] = comp
    
    return df



df = run_all(term)

df[['pos', 'neg']].sum().plot(kind='bar')
plt.show()

pos_ratio = round((df[['pos']].sum().values[0] / np.sum(df[['pos','neg']].sum().values)), 2)
neg_ratio = round((df[['neg']].sum().values[0] / np.sum(df[['pos','neg']].sum().values)), 2)
#df[['pos','neg']].sum().values.sum()

print('Pos - Neg ratio : ',pos_ratio,neg_ratio)

