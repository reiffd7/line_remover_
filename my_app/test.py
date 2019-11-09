import pickle
import pandas as pd
from build_model import TextClassifier, get_data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

X, y = get_data('data/articles.csv')
X1 = ['HOUSTON -- Justin Verlander will follow Gerrit Cole and start Game 2', 'The Astros winning in a sweep is +750 at DraftKings Sportsbook. The Nationals winning in a sweep is 25-1.']


vector = CountVectorizer()
X_train = vector.fit_transform(X)
X_test = vector.transform(X1)
# training_data = vector.transform(X)
bayes = MultinomialNB()
bayes.fit(X_train, y)

y_pred = bayes.predict(X_test)
# print("Accuracy:", model.score(X, y))
# print("Predictions:", model.predict(X1))