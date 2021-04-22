import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, roc_curve, recall_score, precision_score, accuracy_score
from sklearn.inspection import permutation_importance, plot_partial_dependence, partial_dependence



class RandForest():

    def __init__(self, n_estimators, max_depth, max_leaf, max_features, class_weight='balanced'):
        self.forest = RandomForestClassifier(n_estimators=n_estimators, 
                                        max_depth=max_depth, n_jobs=-1, 
                                        max_leaf_nodes=max_leaf, oob_score=True,
                                        class_weight=class_weight)
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.acc = None
        self.oob = None
        self.probas = None
        self.recall = None
        self.precision = None
        self.y_pred = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.forest.fit(X, y)

    def predict(self, X_test, thresh=-1):
        self.X_test = X_test
        probs = self.forest.predict_proba(X_test)
        self.probas = probs

        if thresh > 0:
            y_hat = (probs[:,1] >= thresh).astype('int')
        else: 
            y_hat = self.forest.predict(X_test)

        self.y_pred = y_hat

        return probs, y_hat

    def score(self, y_test, y_hat):
        self.y_test = y_test
        self.oob = self.forest.oob_score_
        self.acc = accuracy_score(y_test, y_hat)
        self.recall = recall_score(y_test, y_hat)
        self.precision = precision_score(y_test, y_hat)

    # def plot_confusion(self):
    #     plot_confusion_matrix(self.forest, self.X_test, self.y_test)

    def plot_confusion(self, labels=['Legit', 'Troll'], ax=None):
    """
        Plot and show a confusion matrix
         Parameters
        -------------------
            true : True Y labels
            pred : Predicted labels
            ax : A matplotlib axis to be plotted, if none, one will be created
         Returns
        -------------------
            None
    """
    # Get Confusion Matrix
    cm = confusion_matrix(self.y_test, self.y_pred)
    # Set up axis
    ax = ax if ax else plt.gca()
    im = ax.imshow(cm)
    ax.set_xticks(cm.shape[0])
    ax.set_yticks(cm.shape[0])
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    # Plot values
    for i in range(cm.shape[1]):
        for j in range(cm.shape[0]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    plt.show()


    # feature importances
        # gini
    def chart_gini_import(self, features=-1, vocab=None):
        fig, ax = plt.subplots()
        
        if features == -1:
            feature_scores = pd.Series(self.forest.feature_importances_, index=list(range(1, self.X.shape[1]+1)))
        else:
            feature_scores = pd.Series(self.forest.feature_importances_, index=vocab[:features]) # NOT CORRECT, NEEED SORTED FEATURES
        
        feature_scores = feature_scores.sort_values()
        ax = feature_scores.plot(kind='barh', figsize=(10,4))
        ax.set_title('Gini Importance')
        ax.set_xlabel('Avg. Contribution to Info Gain');

        # permutations
    def chart_permutation_import(self, n_repeats=5):
        fig, ax = plt.subplots()
        perms = permutation_importance(self.forest, self.X, self.y, n_repeats=n_repeats)
        sorted_idx = perms.importances_mean.argsort()
        ax.boxplot(perms.importances[sorted_idx].T, vert=False, labels=sorted_idx)
        ax.set_title('Permutation Importance');

    


