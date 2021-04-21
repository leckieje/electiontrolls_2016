import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
from src.model import * 
from src.eda_vector import *
import pickle

class LDA_model():

    def __init__(self, topics=5, max_iter=5):
        self.lda = LatentDirichletAllocation(n_components=topics, n_jobs=-1, 
                                             learning_method='online', max_iter=max_iter)
        self.X = None
        self.y = None
        self.vocab = None
        self.fit_model = None
        self.theta_mat = None
        self.phi_mat = None
        self.topic_hood = None
        self.score = None

    def fit(self, X, y, vocab):
        self.X = X
        self.y = y
        self.vocab = vocab
        self.fit_model = self.lda.fit(X)

    def phi(self):
        phi_mat = self.lda.components_
        self.phi_mat = phi_mat
        return phi_mat

    def theta(self):
        theta_mat = self.lda.transform(self.X)
        self.theta_mat = theta_mat
        return theta_mat

    def coherance_score(self, top_n=25):
        score_ = metric_coherence_gensim(measure='u_mass', top_n=top_n, 
                                        topic_word_distrib=self.phi_mat, 
                                        dtm=self.X, vocab=np.array(self.vocab), 
                                        return_coh_model=False, return_mean=True)

        self.score = score_
        return score_ 

    def display_topics(self, num_words=10):
        for topic_idx, topic in enumerate(self.phi_mat):
            print("Topic %d:" % (topic_idx))
            print(" ".join([self.vocab[i]
                            for i in topic.argsort()[:-num_words - 1:-1]]))
    
    def topic_likelihood(self):
        self.topic_hood = np.argmax(self.theta_mat, axis=1)
        return self.topic_hood

    def topics_by_class(self):
        return pd.DataFrame({'topic': self.topic_hood, 'legit': self.y.values})
    
    def plot_topics_by_class(self):
        fig, axs = plt.subplots(2, figsize=(8,10))
        topics = pd.DataFrame({'topic': self.topic_hood, 'legit': self.y.values})
        # split topic likelihood by class
        legit_topics = topics[topics['legit'] == 1] 
        troll_topics = topics[topics['legit'] == 0]
        
        axs[0].hist(legit_topics['topic'], color='blue', label='Legitmate Tweets')
        axs[1].hist(troll_topics['topic'], color='orange', label='Troll Tweets')
        
        axs[0].set_title('Legitmate Tweets by Topic')
        axs[0].set_ylabel('Number of Tweets')
        axs[0].set_xlabel('Topic')
        
        axs[1].set_title('Troll Tweets by Topic')
        axs[1].set_ylabel('Number of Tweets')
        axs[1].set_xlabel('Topic')
        
        fig.tight_layout()
        fig.legend(); 
    
    def plot_top_words(self, n_top_words=10):
        fig, axes = plt.subplots(2, 3, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        
        for topic_idx, topic in enumerate(self.phi_mat):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [self.vocab[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Topic {topic_idx +1}',
                         fontdict={'fontsize': 15})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=20)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle('Top Words by Topic', fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data(num_samples=100000, balanced=True)
    custom_stops = ['https', 'rt', 'co', 'amp', 'via', 'go', 'get', 'said', 'say', 'news', 'new', 'make', 'want', 
                'trump', 'clinton', 'donald', 'donald trump', 'donaldtrump', 'says', 'hillary', 'hillaryclinton',
                'hillary clinton', 'realdonaldtrump', 'would', 'let', 'video', 'like']
    vectors = EDA_vec(X_train, y_train)
    vectors.add_stop_words(custom_stops)
    vocab, matrix = vectors.vectorize()
    model = LDA_model(topics=5)
    model.fit(matrix, y_train, vocab)
    with open('model.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)