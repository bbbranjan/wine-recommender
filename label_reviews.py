import collections
import csv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time
import math

# Load the dataframe
dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews.csv')
print(dataframe['sentiment'])
train_dataframe = dataframe.loc[dataframe['sentiment'] == 'pos']
train_dataframe = train_dataframe.append(dataframe.loc[dataframe['sentiment'] == 'neg'])
test_dataframe = dataframe.drop(train_dataframe.index.values)

print(len(train_dataframe.index))
print(len(test_dataframe.index))
print(len(dataframe.index))

vectorizer = TfidfVectorizer(analyzer='word', strip_accents='ascii', lowercase=True, max_features=10000)
features = vectorizer.fit_transform(train_dataframe['review'])
features_nd = features.toarray() # for easy usage

train_split = 0.6

X_train, X_test, y_train, y_test  = train_test_split(features_nd, train_dataframe['sentiment'], train_size=train_split)



# x_train = features_nd[0:10000]
# x_test = features_nd[10000:]
# y_train = list(dataframe.iloc[0:10000]['sentiment'])
# y_test = list(dataframe.iloc[10000:]['sentiment'])
svc_classifier = LinearSVC()
svc_classifier.fit(X_train, y_train)
svc_y_pred = svc_classifier.predict(X_test)
print(type(y_test), type(svc_y_pred))
print(confusion_matrix(y_test, svc_y_pred))

sentiment_preds = {}

for index, row in test_dataframe.iterrows():
	test_features = vectorizer.transform([row['review']])
	test_features_nd = test_features.toarray() # for easy usage
	dataframe.at[index, 'sentiment'] = svc_classifier.predict(test_features_nd)[0]

dataframe.to_csv('cleaned_wine_reviews_light_sentiment.csv')