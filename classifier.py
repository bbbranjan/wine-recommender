from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import pandas as pd


def main():
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

	train_split = 0.8

	X_train, X_test, y_train, y_test  = train_test_split(features_nd, data_labels, train_size=train_split)
	binary_y_test = [1 if y == 'pos' else 0 for y in y_test]

	# Logistic Regression method
	print('\nLogistic Regression Method')
	print('===========================')
	log_model = LogisticRegression()
	log_model = log_model.fit(X=X_train, y=y_train)
	log_y_pred = log_model.predict(X_test)
	binary_log_y_pred = [1 if y == 'pos' else 0 for y in log_y_pred]
	print('Accuracy Score: {}'.format(accuracy_score(binary_y_test, binary_log_y_pred)))
	print('Cohen Kappa Score: {}'.format(cohen_kappa_score(binary_y_test, binary_log_y_pred)))
	log_precision, log_recall, log_fbeta_score, support = precision_recall_fscore_support(binary_y_test, binary_log_y_pred, average='binary')
	print('Precision Score: {}'.format(log_precision))
	print('Recall Score: {}'.format(log_recall))
	print('F-1 Score: {}'.format(log_fbeta_score))


	# Multinomial Naive Bayes method
	print('\nMultinomial Naive Bayes Method')
	print('===========================')
	nb_classifier = MultinomialNB().fit(X_train, y_train)
	nb_y_pred = nb_classifier.predict(X_test)
	binary_nb_y_pred = [1 if y == 'pos' else 0 for y in nb_y_pred]
	print('Accuracy Score: {}'.format(accuracy_score(binary_y_test, binary_nb_y_pred)))
	print('Cohen Kappa Score: {}'.format(cohen_kappa_score(binary_y_test, binary_nb_y_pred)))
	nb_precision, nb_recall, nb_fbeta_score, support = precision_recall_fscore_support(binary_y_test, binary_nb_y_pred, average='binary')
	print('Precision Score: {}'.format(nb_precision))
	print('Recall Score: {}'.format(nb_recall))
	print('F-1 Score: {}'.format(nb_fbeta_score))


	# Linear SVC method
	print('\nLinear SVC Method')
	print('===========================')
	svc_classifier = LinearSVC()
	svc_classifier.fit(X_train, y_train)
	svc_y_pred = svc_classifier.predict(X_test)
	binary_svc_y_pred = [1 if y == 'pos' else 0 for y in svc_y_pred]
	print('Accuracy Score: {}'.format(accuracy_score(y_test, svc_y_pred)))
	print('Cohen Kappa Score: {}'.format(cohen_kappa_score(y_test, svc_y_pred)))
	svc_precision, svc_recall, svc_fbeta_score, support = precision_recall_fscore_support(binary_y_test, binary_svc_y_pred, average='binary')
	print('Precision Score: {}'.format(svc_precision))
	print('Recall Score: {}'.format(svc_recall))
	print('F-1 Score: {}'.format(svc_fbeta_score))



if __name__ == '__main__':
	main()