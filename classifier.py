from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import pandas as pd

# Load the dataframe
dataframe = pd.read_csv('./wine-reviews/winemag-data_first200.csv')
data_labels = []

for i in range(0, len(dataframe['description'])):
	if dataframe['points'][i] >= 90:
		data_labels.append('pos')
	else:
		data_labels.append('neg')

vectorizer = CountVectorizer(analyzer='word', strip_accents='ascii', lowercase=True)
features = vectorizer.fit_transform(dataframe['description'])
features_nd = features.toarray() # for easy usage

# print('{}'.format(features))

X_train, X_test, y_train, y_test  = train_test_split(features_nd, data_labels, train_size=0.82)

# Logistic Regression method
print('\nLogistic Regression Method')
print('===========================')
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
log_y_pred = log_model.predict(X_test)
print('Accuracy Score: {}'.format(accuracy_score(y_test, log_y_pred)))
print('Cohen Kappa Score: {}'.format(cohen_kappa_score(y_test, log_y_pred)))

# print ('y_test | y_pred')
# for i in range(0, len(X_test)):
# 	print('{0} | {1} '.format(y_test[i], y_pred[i]))


print('\nMultinomial Naive Bayes Method')
print('===========================')
clf = MultinomialNB().fit(X_train, y_train)
nb_y_pred = clf.predict(X_test)
print('Accuracy Score: {}'.format(accuracy_score(y_test, nb_y_pred)))
print('Cohen Kappa Score: {}'.format(cohen_kappa_score(y_test, nb_y_pred)))

print('\nLinear SVC Method')
print('===========================')
classifier = LinearSVC()
classifier.fit(X_train, y_train)
svc_y_pred = classifier.predict(X_test)
print('Accuracy Score: {}'.format(accuracy_score(y_test, svc_y_pred)))
print('Cohen Kappa Score: {}'.format(cohen_kappa_score(y_test, svc_y_pred)))