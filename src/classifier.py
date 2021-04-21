import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance, plot_partial_dependence, partial_dependence



class RandForest():

    def __init__(self, n_estimators, max_depth, max_leaf, max_features):
        self.forest = RandomForestClassifier(n_estimators=n_estimators, 
                                        max_depth=max_depth, n_jobs=-1, 
                                        max_leaf_nodes=max_leaf, oob_score=True)
        self.X = None
        self.y = None
        self.fit_class = None
        self.acc = None
        self.oob = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.fit_class = self.forest.fit(X, y)

    def score(self, X_test, y_test):
        self.acc = self.forest.score(X_test, y_test)
        self.oob = self.forest.oob_score_

    # feature importances
        # gini
    def chart_gini_import(self):
        fig, ax = plt.subplots()
        feature_scores = pd.Series(self.forest.feature_importances_, index=list(range(1, self.X.shape[1]+1)))
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


    


