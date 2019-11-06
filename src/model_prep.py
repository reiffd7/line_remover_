import pandas as pd 
import numpy as np 
import os
import sys
from modeling import Classifiers
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
import pickle
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data') 
CLASSIFICATION_DIRECTORY = os.path.join(DATA_DIRECTORY, 'classification/merged') 
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models/models')


sys.path.append(ROOT_DIRECTORY)






class prepareDF(object):

    def __init__(self, df):
        self.df = df
        self.featurize()
        self.test_train_split(0.25)



    def featurize(self):
        self.y = self.df['label'].to_numpy()
        self.X = self.df.drop(columns = ['label']).to_numpy()
        self.feature_names = self.df.drop(columns = ['label']).columns

    def test_train_split(self, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        





if __name__ == '__main__':

    ## Data Prep
    data_path = os.path.join(CLASSIFICATION_DIRECTORY, 'result.csv')
    df = pd.read_csv(data_path, index_col=0)


    prepare = prepareDF(df)
    # prepare.test_train_split(0.33)

    feature_names = prepare.feature_names
    X_train, X_test, y_train, y_test = prepare.X_train, prepare.X_test, prepare.y_train, prepare.y_test

    print('Done preparing. About to run models ...')
    ## Run Models

    # y_train = y_train.astype(float)
    # X_train = X_train.astype(float)
    param_dist = {"loss": ['deviance', 'exponential'],
                "learning_rate": np.linspace(0.1, 1, 10),
            "n_estimators": [10, 25, 50, 75, 100],
            "subsample": [0.5, 1.0],
              "max_depth": [3, 2, None],
              "max_features": sp_randint(1, 12),
              "min_samples_split": sp_randint(2, 11),
              "criterion": ["friedman_mse", "mse", "mae"]}

    # rf_model = Classifiers(RandomForestClassifier(), param_dist)
    # rf_model.best_parameters(X_train, y_train)

    log_model = Classifiers(LogisticRegression(), param_dist)
    log_model.fit(X_train, y_train)

    log_model.predict(X_test)
    print("Accuracy: {}, Precision: {}, Recall: {}".format(accuracy_score(y_test, log_model.y_pred), precision_score(y_test, log_model.y_pred), recall_score(y_test, log_model.y_pred)))


    filename = os.path.join(MODELS_DIRECTORY, 'barebones.sav')
    pickle.dump(log_model.model, open(filename, 'wb'))



    # rf_best_params = {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 11, 'min_samples_split': 3, 'n_estimators': 25}

    
    # rf_model.fit(X_train, y_train)
    # rf_model.predict(X_test)
    # print("Accuracy: {}, Precision: {}, Recall: {}".format(accuracy_score(y_test, rf_model.y_pred), precision_score(y_test, rf_model.y_pred), recall_score(y_test, rf_model.y_pred)))


    # gbc_best_params = {'criterion': 'friedman_mse', 'learning_rate': 0.9, 'loss': 'exponential', 'max_depth': None, 'max_features': 9, 'min_samples_split': 7, 'n_estimators': 100, 'subsample': 1.0}
    # gbc_model = Classifiers(GradientBoostingClassifier(**gbc_best_params), param_dist)
    # gbc_model.fit(X_train, y_train)
    # gbc_model.predict(X_test)
    # print("Accuracy: {}, Precision: {}, Recall: {}".format(accuracy_score(y_test, gbc_model.y_pred), precision_score(y_test, gbc_model.y_pred), recall_score(y_test, gbc_model.y_pred)))

    # print('Saving Model ...')

        # save the model to disk
    # filename = os.path.join(MODEL_DIRECTORY, 'models/gbc_model1.sav')
    # pickle.dump(gbc_model.model, open(filename, 'wb'))

    # rf_best_params = {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 11, 'min_samples_split': 3, 'n_estimators': 25}
    # gbc_best_params = {'criterion': 'friedman_mse', 'learning_rate': 0.9, 'loss': 'exponential', 'max_depth': None, 'max_features': 9, 'min_samples_split': 7, 'n_estimators': 100, 'subsample': 1.0}
    # models = [Classifiers(LogisticRegression(), param_dist), Classifiers(RandomForestClassifier(**rf_best_params), param_dist), Classifiers(GradientBoostingClassifier(**gbc_best_params), param_dist)]


    # rf = models[2]
    # rf.fit(X_train, y_train)
    # rf.feature_importance(feature_names)
    # rf.partial_dependence(X_train, feature_names)
    # plt.show()
    # results = []
    # for model in models:
    #     model.fit(X_train, y_train)
    #     model.predict(X_test)
    #     model.plot_roc_curve(X_test, y_test)
    #     results.append("{}: Accuracy: {}, Precision: {}, Recall: {}".format(model.model.__class__.__name__, accuracy_score(y_test, model.y_pred), precision_score(y_test, model.y_pred), recall_score(y_test, model.y_pred)))
    # print(results)
    
    # plt.plot(np.linspace(0, 1), np.linspace(0, 1), linestyle = '--', label='Random Guess')
    # plt.title('ROC Curve')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend()
    # plt.show()

