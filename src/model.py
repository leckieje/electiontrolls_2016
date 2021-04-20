import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, words

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation

# generating stop words
custom_stops = []

def get_stop_words(words, custom_stops=custom_stops):
    sw = stopwords.words('english')
    stops = words[0]
    
    for lst in words[1:]:
        stops += lst

    stops = stops + custom_stops
    
    return sw + stops

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