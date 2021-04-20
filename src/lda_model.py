import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim

class LDA_model():

    def __init__(self):
        self.lda = LatentDirichletAllocation(n_components=5, n_jobs=-1, learning_method='online', max_iter=5)
        self.X = None
        self.y = None
        self.fit_model = None
        self.theta = None
        self.phi = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.fit_model = self.lda.fit(X)

    def phi(self):
        self.phi = self.lda.components_
        return self.phi

    def theta(self):
        self.theta = self.lda.transform(self.X)
        return self.theta 

    def coherance_score(self):
        pass

    def display_topics(self, num_top_words):
        for topic_idx, topic in enumerate(self.phi):
            print("Topic %d:" % (topic_idx))
            print(" ".join([self.X.columns[i]
                            for i in topic.argsort()[:-num_top_words - 1:-1]]))
    
    def topic_likelihood(self):
        self.topic_likelihood = np.argmax(self.theta, axis=1)
        return self.topic_likelihood

    def topics_by_class(self):
        return pd.DataFrame({'topic': self.topic_likelihood, 'legit': self.y.values})


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data(num_samples=700000)
    custom_stops = ['https', 'rt']
    stop_words = get_stopwords(custom_stops)
    vocab_count, count_vec = get_countvec(X_train, stop_words=stop_words, min_df=0.005, n_grams=(1,2))
    model = LDA_model()
    model.fit(count_vec, y_train)
    with open('model.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)