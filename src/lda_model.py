import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim

def LDA_model():

    def __init__(self, n_components=5, max_iter=5):
        self.lda = LatentDirichletAllocation(n_components=n_components, n_jobs=-1, 
                                                learning_method='online', max_iter=5)
        self.X = None
        self.y = None
        self.fit = None
        self.theta = None
        self.phi = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.fit = self.lda.fit(self.X)

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
    # X, y = resample_training_data('data/data.json')
    model = LDA_model()
    model.fit(X, y)
    with open('model.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)