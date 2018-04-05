from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import pandas as pd

# Load the dataframe
dataframe = pd.read_csv('./wine-reviews/winemag-data_labelled1500.csv')

data_labels = []

for index, row in dataframe.iterrows():
	if row['points'] >= 85:
		data_labels.append('pos')
	else:
		data_labels.append('neg')

vectorizer = TfidfVectorizer(analyzer='word', strip_accents='ascii', lowercase=True, max_features=10000)
features = vectorizer.fit_transform(dataframe['description'])
features_nd = features.toarray() # for easy usage

# print('{}'.format(features))

train_split = 0.8

X_train, X_test, y_train, y_test  = train_test_split(features_nd, data_labels, train_size=train_split)

# Logistic Regression method
print('\nLogistic Regression Method')
print('===========================')
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
log_y_pred = log_model.predict(X_test)
print('Accuracy Score: {}'.format(accuracy_score(y_test, log_y_pred)))
print('Cohen Kappa Score: {}'.format(cohen_kappa_score(y_test, log_y_pred)))


# Multinomial Naive Bayes method
print('\nMultinomial Naive Bayes Method')
print('===========================')
nb_classifier = MultinomialNB().fit(X_train, y_train)
nb_y_pred = nb_classifier.predict(X_test)
print('Accuracy Score: {}'.format(accuracy_score(y_test, nb_y_pred)))
print('Cohen Kappa Score: {}'.format(cohen_kappa_score(y_test, nb_y_pred)))


# Linear SVC method
print('\nLinear SVC Method')
print('===========================')
svc_classifier = LinearSVC()
svc_classifier.fit(X_train, y_train)
svc_y_pred = svc_classifier.predict(X_test)
print('Accuracy Score: {}'.format(accuracy_score(y_test, svc_y_pred)))
print('Cohen Kappa Score: {}'.format(cohen_kappa_score(y_test, svc_y_pred)))