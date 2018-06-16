from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt


def parseData():
    print("Parsing training data")
    data = pd.read_csv("tweets/semeval/reg_trainSet.tsv", sep='\t')
    y_train = data["class"]
    x_train = data["tweet"]

    print("Parsing test data")
    testdata = pd.read_csv("tweets/semeval/testSet.tsv", sep='\t')
    y_test = testdata["class"]
    x_test = testdata["tweet"]
    
    return x_train, y_train, x_test, y_test

def features(x_train, x_test):
    count_vect = CountVectorizer(max_features = 4000, stop_words=None, ngram_range=(1,2))
    x_train = count_vect.fit_transform(x_train)
    x_test = count_vect.transform(x_test)
    return x_train, x_test

# Baseline algorithm 1
def baseline1(y_train, y_test):
    
    negative_frequency = 0
    neutral_frequency = 0
    positive_frequency = 0
    
    for category in y_train:
        if category == 0:
            negative_frequency += 1
        elif category == 1:
            neutral_frequency += 1
        elif category == 2:
            positive_frequency += 1
    
    if negative_frequency >= neutral_frequency and negative_frequency >= positive_frequency:
        suggested_class = 0
    elif neutral_frequency >= negative_frequency and neutral_frequency >= positive_frequency:
        suggested_class = 1
    elif positive_frequency >= negative_frequency and positive_frequency >= neutral_frequency:
        suggested_class = 2
    
    y_pred = [suggested_class] * len(y_test)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    print (metrics.classification_report(y_test, y_pred))
    
# Baseline algorithm 2
def baseline2(y_train, y_test):
    negative_frequency = 0
    neutral_frequency = 0
    positive_frequency = 0
    
    for category in y_train:
        if category == 0:
            negative_frequency += 1
        elif category == 1:
            neutral_frequency += 1
        elif category == 2:
            positive_frequency += 1
    
    pr_negative = negative_frequency/float(len(y_train))
    pr_neutral = neutral_frequency/float(len(y_train))
    pr_positive = positive_frequency/float(len(y_train))

    y_pred = []
    for i in range(len(y_test)):
        y_pred.append(np.random.choice(np.arange(0, 3), p=[pr_negative, pr_neutral, pr_positive]))
    
    y_pred = np.asarray(y_pred, dtype=np.float32)
    print (metrics.classification_report(y_test, y_pred))

# Building neural network
def MLP(x_train, y_train, x_test, y_test):
    
    clf = MLPClassifier(solver='adam',
                        activation='relu',
                        batch_size=22,
                        max_iter=20,
                        alpha=1e-5,
                        warm_start=True,
                        hidden_layer_sizes=(120,90,60),
                        random_state=1)
                        
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    print metrics.classification_report(y_test, y_pred)
    
    # Learning curves
    y_probas = clf.predict_proba(x_test) 
    skplt.metrics.plot_precision_recall_curve(y_test, y_probas, curves=['each_class'])
    plt.show()
    skplt.metrics.plot_precision_recall_curve(y_test, y_probas, curves=['micro'])
    plt.show()
    skplt.estimators.plot_learning_curve(clf, x_train, y_train)
    plt.show()
    print metrics.classification_report(y_test, y_pred)

def calculateHyperParametersGS(trainFeatures, trainLabels):
    print ("----- CRL Hyper Parameters -----")
    clf = MLPClassifier( activation='relu', batch_size=22, max_iter=20, alpha=1e-5, warm_start=True, hidden_layer_sizes= (120,120,50))

    param_grid = [{#'batch_size': [32, 22],
                   #'warm_start' : [False,True],
                   'random_state':[0,1],
                   #'hidden_layer_sizes' : [(120,120,50),(40,2)]
                   'solver' : ['lbfgs', 'sgd', 'adam']
                 }]

    # search
    rs = GridSearchCV(clf, param_grid,n_jobs=-1,
                           scoring='f1')
    rs.fit(trainFeatures, trainLabels)
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = parseData()

    print("features creation ...")
    x_train, x_test = features(x_train, x_test)
    
    print("Calculating metrics for Baseline 1 ...")
    baseline1(y_train, y_test)
    
    print("Calculating metrics for Baseline 2 ...")
    baseline2(y_train, y_test)

    print("MLP classifier metrics ...")
    MLP(x_train, y_train, x_test, y_test)
#    calculateHyperParametersGS(x_train, y_train)
