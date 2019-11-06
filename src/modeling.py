import pdb
import requests
import warnings
import numpy as np
from sklearn.base import clone
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.datasets import make_classification
from time import time
from scipy.stats import randint as sp_randint
import seaborn as sns




class Classifiers(object):


    def __init__(self, model, param_dist):
        self.param_dist = param_dist
        self.model = model 

    def best_parameters(self, X_train, y_train, n_iter_search=20):
        random_search = RandomizedSearchCV(self.model, 
                                        param_distributions=self.param_dist,
                                        n_iter=n_iter_search, 
                                        scoring = 'precision',
                                        cv=5, 
                                        iid=False)
        start = time()
        random_search.fit(X_train, y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
            " parameter settings." % ((time() - start), n_iter_search))
        print("Best Params: ")
        print(random_search.best_params_)
        
        
        

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        self.y_pred = self.model.predict(X)
        
        
    def feature_importance(self, fnames):
        # if self.model.__class__.__name__ == 'AdaBoostClassifier' or self.model.__class__.__name__ == 'GradientBoostingClassifier' or self.model.__class__.__name__ == 'RandomForestClassifier':
        #     self.feature_importances_ = self.model.feature_importances_

        # if self.model.__class__.__name__ == 'LogisticRegression':
        #     self.feature_importances_ = self.model.coef_

        colindex = np.argsort(self.model.feature_importances_)[::-1]
        self.feature_importances_ = self.model.feature_importances_[colindex]
        self.fnames = fnames[colindex]
        y_ind = np.arange(9, -1, -1) # 9 to 0
        fig = plt.figure(figsize=(8, 8))
        plt.bar(self.fnames, self.feature_importances_, label = self.model.__class__.__name__)
        plt.xticks(rotation=90)
        plt.title('Feature Importance - Random Forest')
        plt.tight_layout()
       

        

    def cross_val(self, X_train, y_train, nfolds):
        ''' Takes an instantiated model (estimator) and returns the average
            mean square error (mse) and coefficient of determination (r2) from
            kfold cross-validation.
            Parameters: estimator: model object
                        X_train: 2d numpy array
                        y_train: 1d numpy array
                        nfolds: the number of folds in the kfold cross-validation
            Returns:  mse: average mean_square_error of model over number of folds
                    r2: average coefficient of determination over number of folds
        
            There are many possible values for scoring parameter in cross_val_score.
            http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            kfold is easily parallelizable, so set n_jobs = -1 in cross_val_score
        '''
        recall = cross_val_score(self.model, X_train, y_train, 
                            scoring='recall',
                            cv=nfolds, n_jobs=-1)
        # mse multiplied by -1 to make positive
        precision = cross_val_score(self.model, X_train, y_train, 
                            scoring='precision', cv=nfolds, n_jobs=-1)

        accuracy = cross_val_score(self.model, X_train, y_train, 
                            scoring='accuracy', cv=nfolds, n_jobs=-1)

        
        accuracy_mean = accuracy.mean()
        precision_mean = precision.mean()
        recall_mean  = recall.mean(0)
        name = self.model.__class__.__name__
        print("{0:<25s} Train CV | Accuracy: {1:0.3f} | Precision: {2:0.3f} | Recall: {3:0.3f}".format(name,
                                                            accuracy_mean, precision_mean, recall_mean))
        return accuracy_mean, precision_mean, recall_mean

    def confusion_matrix(self, y_test):
        self.confusion_matrix = confusion_matrix(y_test, self.y_pred)
        ax= plt.subplot()
        sns.heatmap(self.confusion_matrix, annot=True, ax = ax)
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Not Fraud', 'Fraud'])
        ax.yaxis.set_ticklabels(['Not Fraud', 'Fraud'])
        plt.show()

    def plot_roc_curve(self, x_test, y_test):
        y_scores = self.model.predict_proba(x_test)[::,1]
        # print(y_scores[:, 1])
        # print(y_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        area = round(roc_auc_score(y_test, y_scores), 2)
        plt.plot(fpr, tpr, label = "{} - testing, AUC: {}".format(self.model.__class__.__name__, area))
        
        

    def misclassification_rate(self, y, y_pred):
        '''
        INPUT:
            - y: numpy array, true labels
            - y_pred: numpy array, predicted labels
        '''
        return float((y_pred != y).sum())/len(y)

    def stage_score_plot(self, X_train, X_test, y_train, y_test):
        '''
            Parameters: estimator: GradientBoostingClassifier or 
                                AdaBoostClassifier
                        X_train: 2d numpy array
                        y_train: 1d numpy array
                        X_test: 2d numpy array
                        y_test: 1d numpy array
            Returns: A plot of the number of iterations vs the misclassification rate 
            for the model for both the training set and test set.
        '''

        name = self.model.__class__.__name__.replace('Classifier', '')
        label_str = name
        if "Gradient" in name: 
            md = self.model.max_depth
            label_str += ", max depth: {0}".format(md)
        
        # initialize 
        test_scores = np.zeros((self.model.n_estimators,), dtype=np.float64)
        # Get test score from each boost
        for i, y_test_pred in enumerate(self.model.staged_predict(X_test)):
            test_scores[i] = self.misclassification_rate(y_test, y_test_pred)
        plt.plot(test_scores, alpha=.5, label=label_str, ls = '-', marker = 'o', ms = 3)
        plt.title(self.model.__class__.__name__ + ": Learning Rate = 0.4")
        plt.ylabel("Misclassification rate", fontsize=14)
        plt.xlabel("Iterations", fontsize=14)
        plt.show()

    def partial_dependence(self, X_train, fnames):
        colindex = np.argsort(self.feature_importances_)[::-1]
        plot_partial_dependence(self.model, X_train, colindex,
                                feature_names = fnames,
                                figsize=(12,10))
        plt.title(self.model.__class__.__name__ + " Partial Dependence")
        plt.tight_layout()
        plt.show()

