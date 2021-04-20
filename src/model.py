import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, words

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation

# generate stop words
custom_stops = ['https', 'rt']

def get_stopwords(custom_stops=[]):
    sw = stopwords.words('english')
    return sw + custom_stops

# sklearn count vectorizer
def get_countvec(corpus, stop_words='english', min_df=.01, n_grams=(1,1)):
    vectorizer = CountVectorizer(stop_words=stop_words, min_df=min_df, ngram_range=n_grams)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, X.toarray()

# sklearn tfidf vectorizer
def get_tfidf(corpus, max_features=None, min_df=.01, stop_words='english', n_grams=(1,1)):
    vectorizer = TfidfVectorizer(max_features=None, min_df=min_df, max_df=1.0, stop_words='english', ngram_range=n_grams)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, X.toarray()

# vectorizer to dataframe
def get_dataframe(X, feature_names):
    df = pd.DataFrame(data = X, columns = feature_names)
    return df

# get random samples
def get_random_sample(df, num_samples):
    samp_idx = np.random.choice(range(len(df)), size=num_samples, replace=False)
    df_samp = df.iloc[samp_idx, :]
    return df_samp

# load and split data
def get_data(num_samples=700000):
    # load data
    legit = pd.read_csv('data/legit_tweets.csv', parse_dates = ['date'])
    troll = pd.read_csv('data/troll_tweets.csv', parse_dates = ['date'])
    legit['legit'] = 1

    # limit troll timeframe
    troll_summer = troll[(troll['date'] >= '2016-06-28') & (troll['date'] <= '2016-11-02')]

    # get samples
    troll_samp = get_random_sample(troll_summer, num_samples)
    legit_samp = get_random_sample(legit, num_samples)

    # combine legit and troll tweets
    total_tweets = pd.concat([legit_samp.loc[:,['text','legit']], troll_samp.loc[:,['text','legit']]])
    total_tweets.reset_index(drop=True, inplace=True)

    # set X and y, split
    X = total_tweets['text']
    y = total_tweets['legit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    return X_train, X_test, y_train, y_test
