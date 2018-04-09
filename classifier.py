from gensim.models import word2vec

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import pandas as pd
import numpy as np
import re
import string
import time

methods = []
accuracy_scores = []
inter_annotator_agreement_scores = []
precision_scores = []
recall_scores = []
fbeta_scores = []
prediction_times = []

def remove_punctuation(x):
    # Lowercases all words
    x = x.lower()
    # Removes non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removes (replaces with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)

def tfidfClassification(dataframe, data_labels, train_split):

	start_fit = time.time()
	vectorizer = TfidfVectorizer(analyzer='word', strip_accents='ascii', lowercase=True, max_features=10000)
	features = vectorizer.fit_transform(dataframe['review'])
	features_nd = features.toarray() # for easy usage
	print('Time taken to fit data: {:0.4f}'.format(time.time() - start_fit))

	X_train, X_test, y_train, y_test  = train_test_split(features_nd, data_labels.values(), train_size=train_split)

	print(len(X_train), len(X_test))

	# Logistic Regression method
	methods.append('Logistic Regression')
	start_log = time.time()
	log_model = LogisticRegression()
	log_model = log_model.fit(X=X_train, y=y_train)
	log_y_pred = log_model.predict(X_test)
	prediction_times.append(float('{:0.4f}'.format(time.time() - start_log)))
	accuracy_scores.append(float('{:0.2f}'.format(accuracy_score(y_test, log_y_pred))))
	inter_annotator_agreement_scores.append(float('{:0.2f}'.format(cohen_kappa_score(y_test, log_y_pred))))
	log_precision, log_recall, log_fbeta_score, _ = precision_recall_fscore_support(y_test, log_y_pred, average='macro')
	precision_scores.append(float('{:0.2f}'.format(log_precision)))
	recall_scores.append(float('{:0.2f}'.format(log_recall)))
	fbeta_scores.append(float('{:0.2f}'.format(log_fbeta_score)))


	# Multinomial Naive Bayes method
	methods.append('Multinomial Naive Bayes')
	start_nb = time.time()
	nb_classifier = MultinomialNB().fit(X_train, y_train)
	nb_y_pred = nb_classifier.predict(X_test)
	prediction_times.append(float('{:0.4f}'.format(time.time() - start_nb)))
	accuracy_scores.append(float('{:0.2f}'.format(accuracy_score(y_test, nb_y_pred))))
	inter_annotator_agreement_scores.append(float('{:0.2f}'.format(cohen_kappa_score(y_test, nb_y_pred))))
	nb_precision, nb_recall, nb_fbeta_score, _ = precision_recall_fscore_support(y_test, nb_y_pred, average='macro')
	precision_scores.append(float('{:0.2f}'.format(nb_precision)))
	recall_scores.append(float('{:0.2f}'.format(nb_recall)))
	fbeta_scores.append(float('{:0.2f}'.format(nb_fbeta_score)))


	# Linear SVC method
	methods.append('Linear SVC')
	start_svc = time.time()
	svc_classifier = LinearSVC()
	svc_classifier.fit(X_train, y_train)
	svc_y_pred = svc_classifier.predict(X_test)
	prediction_times.append(float('{:0.4f}'.format(time.time() - start_svc)))
	accuracy_scores.append(float('{:0.2f}'.format(accuracy_score(y_test, svc_y_pred))))
	inter_annotator_agreement_scores.append(float('{:0.2f}'.format(cohen_kappa_score(y_test, svc_y_pred))))
	svc_precision, svc_recall, svc_fbeta_score, _ = precision_recall_fscore_support(y_test, svc_y_pred, average='macro')
	precision_scores.append(float('{:0.2f}'.format(svc_precision)))
	recall_scores.append(float('{:0.2f}'.format(svc_recall)))
	fbeta_scores.append(float('{:0.2f}'.format(svc_fbeta_score)))


# Function to average all word vectors in a paragraph
def getFeatureVec(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    feature_vec = np.zeros(num_features,dtype="float32")
    word_count = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    i2w_set = set(model.wv.index2word)
    
    for word in  words:
        if word in i2w_set:
            word_count = word_count + 1
            feature_vec = np.add(feature_vec,model[word])
    
    # Not dividing the result by number of words to get average
    # feature_vec = np.divide(feature_vec, word_count)
    return feature_vec

# Function for calculating the average feature vector
def getDocFeatureVecs(reviews, model, num_features):
    counter = 0
    doc_feature_vecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        doc_feature_vecs[counter] = getFeatureVec(review, model, num_features)
        counter = counter+1
        
    return doc_feature_vecs

def Word2VecClassification(dataframe, data_labels, train_split):

	# Creating the model and setting values for the various parameters
	num_features = 300  # Word vector dimensionality
	min_word_count = 40 # Minimum word count
	num_workers = 4     # Number of parallel threads
	context = 10        # Context window size
	downsampling = 1e-3 # (0.001) Downsample setting for frequent words
	
	model = word2vec.Word2Vec(dataframe['review'],\
	                          workers=num_workers,\
	                          size=num_features,\
	                          min_count=min_word_count,\
	                          window=context,
	                          sample=downsampling)

	# To make the model memory efficient
	model.init_sims(replace=True)

	# Saving the model for later use. Can be loaded using Word2Vec.load()
	model_name = "labelled1500_light_300features_40minwords_10context"
	model.save(model_name)

	train_reviews = dataframe.sample(frac=train_split)
	train_data_labels = []
	for index, row in train_reviews.iterrows():
		train_data_labels.append(data_labels[index])
	train_data_vecs = getDocFeatureVecs(train_reviews['review'], model, num_features)

	print(len(train_reviews['review']))

	dataframe = dataframe.drop(train_reviews.index.values)

	test_reviews = dataframe
	test_data_labels = []

	print(len(test_reviews['review']))

	for index, row in test_reviews.iterrows():
		test_data_labels.append(data_labels[index])
	test_data_vecs = getDocFeatureVecs(test_reviews['review'], model, num_features)

	# Random Forest method
	methods.append('Random Forest')
	start_rf = time.time()
	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(train_data_vecs, train_data_labels)
	predicted_labels = forest.predict(test_data_vecs)
	prediction_times.append(float('{:0.4f}'.format(time.time() - start_rf)))
	# predicted_labels = result

	accuracy_scores.append(float('{:0.2f}'.format(accuracy_score(test_data_labels, predicted_labels))))
	inter_annotator_agreement_scores.append(float('{:0.2f}'.format(cohen_kappa_score(test_data_labels, predicted_labels))))
	rf_precision, rf_recall, rf_fbeta_score, _ = precision_recall_fscore_support(test_data_labels, predicted_labels, average='macro')
	precision_scores.append(float('{:0.2f}'.format(rf_precision)))
	recall_scores.append(float('{:0.2f}'.format(rf_recall)))
	fbeta_scores.append(float('{:0.2f}'.format(rf_fbeta_score)))

def main():
	"""
	Main method.
	"""

	# Load the dataframe
	dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews_light.csv')
	neg_rows = int(0.1*len(dataframe["review"])/2)
	pos_rows = int(0.1*len(dataframe["review"])/2)
	data_labels = {}
	for index, row in dataframe.iterrows():
		if row['rating'] > 85:
			if pos_rows == 0:
				dataframe = dataframe.drop(index)
			else:
				pos_rows = pos_rows - 1
				data_labels[index] = 'pos'
		else:
			if neg_rows == 0:
				dataframe = dataframe.drop(index)
			else:
				neg_rows = neg_rows - 1
				data_labels[index] = 'neg'
			

	# Remove punctuation
	# dataframe["review"] = dataframe["review"].map(remove_punctuation)

	print(neg_rows, pos_rows, len(dataframe["review"]))

	train_split = 0.6
	print('\n\nTraining split: {}%\n\n'.format(float(train_split*100)))
	tfidfClassification(dataframe, data_labels, train_split)
	Word2VecClassification(dataframe, data_labels, train_split)

	print('| Classification Method | Accuracy | Kappa | Precision | Recall | F-score | Time')
	for i in range(0, len(methods)):
		print('| {0} | {1} | {2} | {3} | {4} | {5} | {6}'.format(methods[i], accuracy_scores[i], inter_annotator_agreement_scores[i], precision_scores[i], recall_scores[i], fbeta_scores[i], prediction_times[i]))

	# train_split = 0.7
	# print('\n\nTraining split: {}%\n\n'.format(float(train_split*100)))
	# tfidfClassification(dataframe, data_labels, train_split)
	# Word2VecClassification(dataframe, data_labels, train_split)

	# train_split = 0.8
	# print('\n\nTraining split: {}%\n\n'.format(float(train_split*100)))
	# tfidfClassification(dataframe, data_labels, train_split)
	# Word2VecClassification(dataframe, data_labels, train_split)


if __name__ == '__main__':
	main()