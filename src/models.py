import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
import os
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
import seaborn as sns




def score(obj, X_test, y_test):
    name = str(obj.__class__).split(".")[-1][:-2]
    score = obj.score(X_test,y_test)
    y_pred = obj.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    precision = tp / (tp +fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return score


def score_matrix(obj, X_test, y_test):
    name = str(obj.__class__).split(".")[-1][:-2]
    score = obj.score(X_test,y_test)
    y_pred = obj.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    precision = tp / (tp +fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("Model: {}".format(name))
    print("precision = {:.2%}".format(precision))
    print("recall = {:.2%}".format(recall))
    print('accuracy = {:.2%}, score = {:.2%}'.format(accuracy,score))



def cross_val(model, X_train, y_train, k):
    kf = KFold(n_splits = k, shuffle = True)
    
    scores = []
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    for train, test in kf.split(X_train):
        
        model.fit(X_train[train], y_train[train])
        scores.append(score(model, X_train[test], y_train[test]))
        
    return 'Cross Validation Score', np.mean(scores)



def gridsearch(model, parameters, X_train, y_train, X_test, y_test):                         
    scorer = make_scorer(log_loss,
                         greater_is_better=False,
                         needs_proba=True)
    clf = GridSearchCV(model,
                       parameters,
                       cv=10,
                       scoring=scorer)
    clf.fit(X_train,y_train)
    print(f"log loss = {-clf.score(X_test, y_test)}")
    print(f"accuracy = {(clf.predict(X_test) == y_test).mean()}")
    print('Best Estimator: \n ', clf.best_estimator_)
    print('Best Params: \n', clf.best_params_)



def model_score(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    print('Score Matrix: ')
    print(score_matrix(model, X_test, y_test), '\n')
    print('Classification Report: \n')
    print(classification_report(y_test, predict))



def feature_shuffle(rf, X, y):
    names = X.columns
    scores = defaultdict(list)
 
    # crossvalidate the scores on a number of 
    # different random splits of the data
    splitter = ShuffleSplit(100, test_size=.3)

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.values[train_idx], X.values[test_idx]
        y_train, y_test = y.values[train_idx], y.values[test_idx]
        rf.fit(X_train, y_train)
        acc = r2_score(y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(y_test, rf.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)

    score_series = pd.DataFrame(scores).mean()
    scores = pd.DataFrame({'Mean Decrease Accuracy' : score_series})
    scores.sort_values(by='Mean Decrease Accuracy').plot(kind='barh',figsize=(9,9))



def confusion(cm):
    ax= plt.subplot()
    labels =  np.array([[633, 202],
           [293, 492]])
    sns.heatmap(cm, annot=True, ax = ax, fmt='g', cbar=False, annot_kws={"size": 12}, cmap='Greens'); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=15);ax.set_ylabel('Actual', fontsize=15); 
    ax.set_title('Confusion Matrix', fontsize=20); 
    ax.yaxis.set_ticklabels(['Not Churn', 'Churn'], fontsize=12); ax.xaxis.set_ticklabels(['Not Churn', 'Churn'], fontsize=12);



