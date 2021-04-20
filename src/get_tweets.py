import pandas as pd
import tweepy
import csv
import time

# Twitter Developer keys here
consumer_key = 'XXX'
consumer_key_secret = 'XXX'
access_token = 'XXX'
access_token_secret = 'XXX'

# 1. authorization of consumer key and consumer secret
# 2. set access to user's access key and access secret
# 3. calling the api 
auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# this functioin will pull and parse tweets but is prohibitively slow 
def get_tweets(ids, df=True):
    
    tweet_d = {
        'date': [],
        'user': [],
        'text': [],
        'location': [],
        'retweets': [],
        'favs': []
    }
    
    excepts = []
    idx = -1
    
    for i in ids:
        idx += 1
        time.sleep(1)
        try:
            status = api.get_status(i)
            tweet_d['date'].append(status.created_at)
            tweet_d['text'].append(status.text)
            tweet_d['user'].append(status.user.screen_name)
            tweet_d['location'].append(status.user.location)
            tweet_d['retweets'].append(status.retweet_count)
            tweet_d['favs'].append(status.favorite_count)
        except:
            print(f"Exception - idx: {idx}")
            excepts.append(i)
            continue
        
    if df:
        return pd.DataFrame(tweet_d), excepts
    else:
        return tweet_d, excepts