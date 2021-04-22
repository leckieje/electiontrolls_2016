import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, words

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


class EDA_vec():

    def __init__(self, X, y):
        self.X = X
        self.y = y.values
        self.troll_count = sum(y)
        self.legit_count = len(y) - sum(y)
        self.legit_dist = (len(y) - sum(y)) / len(y)
        self.troll_dist = sum(y) / len(y)
        self.troll_freq = None
        self.legit_freq = None
        self.freq_diff = None
        self.stop_words = stopwords.words('english')
        self.vocab = None 
        self.vec_fit = None
        self.vec_shape = None
        self.vec_data = None

    def add_stop_words(self, custom_stops):
        sw = self.stop_words + custom_stops
        self.stop_words = sw

    def vectorize(self, vec_type='count', min_df=.000001, max_df=1.0, n_grams=(1,2), max_features=None):
        if vec_type == 'count':
            vectorizer = CountVectorizer(stop_words=self.stop_words, min_df=min_df, ngram_range=n_grams)
        elif vec_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, stop_words=self.stop_words, ngram_range=n_grams)
        else:
            print('Please specify a vector type')

        self.vec_fit = vectorizer.fit(self.X)
        self.vocab = vectorizer.get_feature_names()
    
        return self.vocab, self.vec_fit

    # diff in word frequenncy ## TO DO: VEC_DATA NO LONGER EXISTS. REPLACE BELOW!!!!
    def word_freq(self):
        df = pd.DataFrame(data=self.vec_data.toarray(), columns=self.vocab)
        df['legit'] = self.y
        legit = df[df['legit'] == 0]
        troll = df[df['legit'] == 1]
        legit_fr = legit.sum().apply(lambda x: x/len(legit))
        troll_fr = troll.sum().apply(lambda x: x/len(troll))
        self.legit_freq = legit_fr.sort_values(ascending=False)
        self.troll_freq = troll_fr.sort_values(ascending=False)

    def chart_word_freq(self, word_lst=None, low_words=0, high_words=5):
        fig, ax = plt.subplots()
        
        df = pd.concat([self.troll_freq, self.legit_freq,], axis=1)
        df['diff'] = np.abs(df[0] - df[1])
        df.sort_values(by='diff', ascending=False, inplace=True)
        self.freq_diff = df
        
        if not word_lst:
            word_lst = df.index[low_words+1:high_words+1]
        
        labels = word_lst 
        legi = df.loc[word_lst, 1]
        trol = df.loc[word_lst, 0]
        
        x = np.arange(len(labels))
        width = 0.35
        legit_bar = ax.bar(x-width/2, legi, width, color='blue', label='Legit')
        troll_bar = ax.bar(x+width/2, trol, width, color='orange', label='Troll')
        
        ax.set_ylabel('Frequency')
        ax.set_title('Word Use Frequency')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout();
    
    # top words/ class
    def word_freq_df(self):
        df = pd.concat([self.troll_freq, self.legit_freq,], axis=1)
        df['diff'] = np.abs(df[0] - df[1])
        df.sort_values(by='diff', ascending=False, inplace=True)
        
        return df
    
    def top_words(self, low=0, high=5):
        legit_words = self.legit_freq.index[low+1:high+1]
        troll_words = self.troll_freq.index[low+1:high+1]
        
        return troll_words, legit_words
    
    def chart_top_words(self, low=0, high=6):
        fig, ax = plt.subplots(2)
        legit_words = self.legit_freq.index[low+1:high+1]
        troll_words = self.troll_freq.index[low+1:high+1]
        
        x = np.arange(len(troll_words))
        ax[0].bar(x, self.legit_freq[legit_words])
        ax[1].bar(x, self.troll_freq[troll_words])
        
        ax[0].set_title('Top Words: Legit')
        ax[0].set_ylabel('Frequency')
        ax[0].set_xticklabels(legit_words)
        
        ax[1].set_title('Top Words: Trolls')
        ax[1].set_ylabel('Frequency')
        ax[1].set_xticklabels(troll_words)
        
        fig.tight_layout()

    # vectorizer to dataframe
    def _get_dataframe(self):
        return pd.DataFrame(data=self.vec_data.toarray(), columns=self.vocab)
        
