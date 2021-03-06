import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score, plot_confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from src.lda_model import *
from src.classifier import *


# COHERANCE SCORE
# test for best num_topics:
def test_topics(X, y, vocab, low, high, by):
    scores = []
    
    for n_topic in range(low, high, by):
        lda = LDA_model(topics=n_topic)
        lda.fit(X, y, vocab)
        lda.phi()
        score = lda.coherance_score()
        scores.append((n_topic, score))
        
    return scores

def print_topic_score(scores):
    for score in scores:
        print(f'#{score[0]} --> {score[1]}')
        
def plot_topic_scores(scores):
    fig, ax = plt.subplots()
    x = [n_topics[0] for n_topics in scores]
    y = [score[1] for score in scores]
    
    ax.plot(x, y)
    ax.set_title('Coherence Score by Topic')
    ax.set_xlabel('# of Topics')
    ax.set_ylabel('Coherence Score');


# RANDOM FOREST

# Random Forest Evaluation, X = theta(w/LDA) or matrix(w/o LDA), y = y_train
def eval_random_forest(X, y, folds=15, n_estimators=500, max_depth=15, max_leaf=None, thresh=-1):
    
    kf = KFold(n_splits=folds, shuffle=True)
    accur = []
    oob_  = []
    rec   = []
    prec  = []
    f_1   = []
    iters = 0
    
    for train, test in kf.split(X):
        
        # random forest 
        forest = RandForest(n_estimators=n_estimators, max_depth=max_depth, 
                            max_leaf=max_leaf, max_features=theta.shape[1])
        forest.fit(X[train], y.iloc[train])
        probas, y_hat = forest.predict(X[test], thresh=thresh)
        forest.score(y.iloc[test], y_hat)
        
        accur.append(forest.acc)
        oob_.append(forest.oob)
        rec.append(forest.recall)
        prec.append(forest.precision)
        f_1.append(forest.f_one)

        iters += 1
        if iters % 5 == 0:
            print(iters)
        
    
    return np.mean(accur), np.mean(oob_), np.mean(rec), np.mean(prec), np.mean(f_1), forest

# ROC curve
def plot_roc_curve(X, y, lda=False, n_estimators=1000, max_depth=100, max_leaf=None):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, 
                                                        shuffle=True, stratify=y)
    fig, ax = plt.subplots()
    
    model = RandForest(n_estimators=n_estimators, max_depth=max_depth, 
                            max_leaf=max_leaf, max_features=X.shape[1])
    model.fit(X_train, y_train)
    probas, y_hat = model.predict(X_test, thresh=-1)
    fpr, tpr, thresh = roc_curve(y_test, probas[:,1])
    
    ax.plot(fpr, tpr)
    ax.plot([0,1], [0,1], ls='--', color='k')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    if lda:
        ax.set_title(f'Random Forest ROC Curve - with LDA')
    else:
        ax.set_title(f'Random Forest ROC Curve - w/out LDA')
    
    return [fpr, tpr, thresh]

# Precision/Recall Curve
def plot_prec_roc_curve(X, y, lda=False, n_estimators=1000, max_depth=100, max_leaf=None, thresh=-1):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, 
                                                        shuffle=True, stratify=y)
    fig, ax = plt.subplots()
    
    model = RandForest(n_estimators=n_estimators, max_depth=max_depth, 
                            max_leaf=max_leaf, max_features=X.shape[1])
    model.fit(X_train, y_train)
    probas, y_hat = model.predict(X_test, thresh=thresh)
    precision, recall, thresh = precision_recall_curve(y_test, probas[:,1])
    
    ax.plot(precision, recall)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    if lda:
        ax.set_title(f'Precision/Recall Curve - with LDA')
    else:
        ax.set_title(f'Precision/Recall Curve - w/out LDA')
    
    return [precision, recall, thresh]

# Parameter testing
def test_forest_depth(X, y, depth_lst, thresh=.5, max_features='log2', lda=False):
    accur = []
    oob_  = []
    rec   = []
    prec  = []
    f_1   = []

    for depth in depth_lst:
        forest = RandForest(n_estimators=est, max_depth=depth, max_leaf=100, 
                            max_features=max_features, class_weight='balanced')
        forest.fit(X, y)
        probas, y_hat = forest.predict(matrix_test, thresh=th_hold)
        forest.score(y_test, y_hat)
        accur.append(forest.acc)
        oob_.append(forest.oob)
        rec.append(forest.recal)
        prec.append(forest.precision)
        f_1.append(forest.f_one)
        
        if lda:
            print('Random Forest Scores -- w/ LDA')
        else:
            print('Random Forest Scores -- w/out LDA')

        print(f'Depth = {depth}')
        print(f'accuracy = {forest.acc}')
        print(f'oob = {forest.oob}')
        print(f'recall = {forest.recall}')
        print(f'precision = {forest.precision}')
        print(f'f_1 score = {forest.f_one}')
        print('-----------------')
    
    return [accur, oob_, rec, prec, f_1]

def test_forest_estimators(X, y, est_lst, thresh=.5, max_features='log2', lda=False):
    accur = []
    oob_  = []
    rec   = []
    prec  = []
    f_1   = []

    for est in est_lst:
        forest = RandForest(n_estimators=est, max_depth=100, max_leaf=100, 
                            max_features=max_features, class_weight='balanced')
        forest.fit(X, y)
        probas, y_hat = forest.predict(matrix_test, thresh=th_hold)
        forest.score(y_test, y_hat)
        accur.append(forest.acc)
        oob_.append(forest.oob)
        rec.append(forest.recal)
        prec.append(forest.precision)
        f_1.append(forest.f_one)
        
        if lda:
            print('Random Forest Scores -- w/ LDA')
        else:
            print('Random Forest Scores -- w/out LDA')

        print(f'Estimators = {est}')
        print(f'accuracy = {forest.acc}')
        print(f'oob = {forest.oob}')
        print(f'recall = {forest.recall}')
        print(f'precision = {forest.precision}')
        print(f'f_1 score = {forest.f_one}')
        print('-----------------')
    
    return [accur, oob_, rec, prec, f_1]

def test_max_leafs(X, y, leaf_lst):
    accuracy = []
    for leaf in leaf_lst:
        accuracy.append(eval_random_forest(X, y, max_leaf=leaf))
        
    return accuracy

def plot_folds_random_forest_folds(X, y, fold_lst):
    fig, ax = plt.subplots()
    xs = fold_lst
    ys = test_forest_folds(X, y, fold_lst)
    ax.plot(xs, ys)
    ax.set_title('Random Forest Accuracy by Folds')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Folds');
    
def plot_depth_random_forest(X, y, depth_lst):
    fig, ax = plt.subplots()
    xs = depth_lst
    ys = test_forest_depth(X, y, depth_lst)
    ax.plot(xs, ys)
    ax.set_title('Random Forest Accuracy by Depth')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Depth');