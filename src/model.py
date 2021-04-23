import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# generate stop words
custom_stops = ['https', 'rt', 'co', 'amp', 'via', 'go', 'get', 'said', 'say', 'news', 'new', 'make', 'want', 
                'trump', 'clinton', 'donald', 'donald trump', 'donaldtrump', 'says', 'hillary', 'hillaryclinton',
                'hillary clinton', 'realdonaldtrump', 'would', 'let', 'video', 'like']

# clean tweets hydrated from GWU
def clean_hydrated(filepath):  
    df = pd.read_csv(filepath)
    df = df[df['lang'] == 'en']
    df = df.loc[:, ['created_at', 'user_screen_name', 'text', 'user_location', 'retweet_count', 'favorite_count']]
    df.rename(columns={'created_at': 'date', 'user_screen_name': 'user', 'text': 'text', 'user_location': 'location', 
                       'retweet_count': 'retweets', 'favorite_count': 'favs'}, inplace=True)
    df.drop_duplicates(ignore_index=True, inplace=True)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df['legit'] = 1
    
    return df

# get random samples
def get_random_sample(df, num_samples, replace=False):
    rand = np.random.RandomState(55)
    samp_idx = rand.choice(range(len(df)), size=num_samples, replace=replace)
    df_samp = df.iloc[samp_idx, :]
    return df_samp

# load and split data
def get_data():
    # load data
    legit = pd.read_csv('data/legit_tweets.csv', parse_dates = ['date'])
    troll = pd.read_csv('data/troll_tweets.csv', parse_dates = ['date'])
    troll['legit'] = 1
    legit['legit'] = 0

    # limit troll timeframe
    troll_summer = troll[(troll['date'] >= '2016-06-28') & (troll['date'] <= '2016-11-02')]

    # combine legit and troll tweets
    total_tweets = pd.concat([legit.loc[:,['text','legit']], troll_summer.loc[:,['text','legit']]])
    total_tweets.reset_index(drop=True, inplace=True)

    # set X and y
    X = total_tweets['text']
    y = total_tweets['legit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=45)

    return X_train, X_test, y_train, y_test

def get_samples(X_train, X_test, y_train, y_test, samp_size=0.1):

    train = pd.DataFrame({'text': X_train, 'legit': y_train})
    test = pd.DataFrame({'text': X_test, 'legit': y_test})

    train = get_random_sample(train, int(len(train)*samp_size))
    test = get_random_sample(test, int(len(test)*samp_size))

    train_Xout = train['text']
    train_yout = train['legit']

    test_Xout = test['text']
    test_yout = test['legit']

    return train_Xout, test_Xout, train_yout, test_yout

def get_data_wrapper(sample=True, samp_size=0.1):

    X_train, X_test, y_train, y_test = get_data()

    if sample:
        train_Xout, test_Xout, train_yout, test_yout = get_samples(X_train, X_test, y_train, y_test, samp_size=samp_size)
        return train_Xout, test_Xout, train_yout, test_yout
    
    else:
        return X_train, X_test, y_train, y_test