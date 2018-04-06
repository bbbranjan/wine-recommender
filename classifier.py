from gensim.models import word2vec

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import pandas as pd
import numpy as np
import time

def tfidfClassification(train_split):

	# Load the dataframe
	dataframe = pd.read_csv('./wine-reviews/winemag-data_labelled1500.csv')

	data_labels = []

	for index, row in dataframe.iterrows():
		if row['points'] >= 85:
			data_labels.append('pos')
		else:
			data_labels.append('neg')

	start_fit = time.time()
	vectorizer = TfidfVectorizer(analyzer='word', strip_accents='ascii', lowercase=True, max_features=10000)
	features = vectorizer.fit_transform(dataframe['description'])
	features_nd = features.toarray() # for easy usage
	print('Time taken to fit data: {}'.format(time.time() - start_fit))

	X_train, X_test, y_train, y_test  = train_test_split(features_nd, data_labels, train_size=train_split)
	binary_y_test = [1 if y == 'pos' else 0 for y in y_test]

	# Logistic Regression method
	print('\nLogistic Regression Method')
	print('===========================')
	start_log = time.time()
	log_model = LogisticRegression()
	log_model = log_model.fit(X=X_train, y=y_train)
	log_y_pred = log_model.predict(X_test)
	print('Time taken for Logistic Regression to predict {0}% data: {1}'.format((1-train_split)*100,time.time() - start_log))
	binary_log_y_pred = [1 if y == 'pos' else 0 for y in log_y_pred]
	print('Accuracy Score: {}'.format(accuracy_score(binary_y_test, binary_log_y_pred)))
	print('Cohen Kappa Score: {}'.format(cohen_kappa_score(binary_y_test, binary_log_y_pred)))
	log_precision, log_recall, log_fbeta_score, _ = precision_recall_fscore_support(binary_y_test, binary_log_y_pred, average='binary')
	print('Precision Score: {}'.format(log_precision))
	print('Recall Score: {}'.format(log_recall))
	print('F-1 Score: {}'.format(log_fbeta_score))


	# Multinomial Naive Bayes method
	print('\nMultinomial Naive Bayes Method')
	print('===========================')
	start_nb = time.time()
	nb_classifier = MultinomialNB().fit(X_train, y_train)
	nb_y_pred = nb_classifier.predict(X_test)
	print('Time taken for Multinomial NB to predict {0}% data: {1}'.format((1-train_split)*100,time.time() - start_nb))
	binary_nb_y_pred = [1 if y == 'pos' else 0 for y in nb_y_pred]
	print('Accuracy Score: {}'.format(accuracy_score(binary_y_test, binary_nb_y_pred)))
	print('Cohen Kappa Score: {}'.format(cohen_kappa_score(binary_y_test, binary_nb_y_pred)))
	nb_precision, nb_recall, nb_fbeta_score, _ = precision_recall_fscore_support(binary_y_test, binary_nb_y_pred, average='binary')
	print('Precision Score: {}'.format(nb_precision))
	print('Recall Score: {}'.format(nb_recall))
	print('F-1 Score: {}'.format(nb_fbeta_score))


	# Linear SVC method
	print('\nLinear SVC Method')
	print('===========================')
	start_svc = time.time()
	svc_classifier = LinearSVC()
	svc_classifier.fit(X_train, y_train)
	svc_y_pred = svc_classifier.predict(X_test)
	print('Time taken for Linear SVC to predict {0}% data: {1}'.format((1-train_split)*100,time.time() - start_svc))
	binary_svc_y_pred = [1 if y == 'pos' else 0 for y in svc_y_pred]
	print('Accuracy Score: {}'.format(accuracy_score(y_test, svc_y_pred)))
	print('Cohen Kappa Score: {}'.format(cohen_kappa_score(y_test, svc_y_pred)))
	svc_precision, svc_recall, svc_fbeta_score, _ = precision_recall_fscore_support(binary_y_test, binary_svc_y_pred, average='binary')
	print('Precision Score: {}'.format(svc_precision))
	print('Recall Score: {}'.format(svc_recall))
	print('F-1 Score: {}'.format(svc_fbeta_score))


# Function to average all word vectors in a paragraph
def getFeatureVec(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    feature_vec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    i2w_set = set(model.wv.index2word)
    
    for word in  words:
        if word in i2w_set:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec,model[word])
    
    # Dividing the result by number of words to get average
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    avg_feature_vecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        avg_feature_vecs[counter] = getFeatureVec(review, model, num_features)
        counter = counter+1
        
    return avg_feature_vecs

def Word2VecClassification(train_split):
	# Load the dataframe
	dataframe = pd.read_csv('./wine-reviews/winemag-data_labelled1500_light.csv')

	data_labels = []

	for index, row in dataframe.iterrows():
		if row['points'] >= 85:
			data_labels.append('pos')
		else:
			data_labels.append('neg')

	# Creating the model and setting values for the various parameters
	num_features = 300  # Word vector dimensionality
	min_word_count = 40 # Minimum word count
	num_workers = 4     # Number of parallel threads
	context = 10        # Context window size
	downsampling = 1e-3 # (0.001) Downsample setting for frequent words
	
	model = word2vec.Word2Vec(dataframe['description'],\
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
	train_data_vecs = getAvgFeatureVecs(train_reviews['description'], model, num_features)

	test_reviews = dataframe.sample(frac=(1-train_split))
	test_data_labels = []

	for index, row in test_reviews.iterrows():
		if data_labels[index] == 'pos':
			test_data_labels.append(1)
		else:
			test_data_labels.append(0)
	test_data_vecs = getAvgFeatureVecs(test_reviews['description'], model, num_features)

	# Random Forest method
	print('\nRandom Forest Method')
	print('===========================')
	start_rf = time.time()
	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(train_data_vecs, train_data_labels)
	result = forest.predict(test_data_vecs)
	print('Time taken for Random Forest to predict {0}% data: {1}'.format((1-train_split)*100,time.time() - start_rf))
	predicted_labels = []
	for label in result:
		if label == 'pos':
			predicted_labels.append(1)
		else:
			predicted_labels.append(0)

	print('Accuracy Score: {}'.format(accuracy_score(test_data_labels, predicted_labels)))
	print('Cohen Kappa Score: {}'.format(cohen_kappa_score(test_data_labels, predicted_labels)))
	rf_precision, rf_recall, rf_fbeta_score, _ = precision_recall_fscore_support(test_data_labels, predicted_labels, average='binary')
	print('Precision Score: {}'.format(rf_precision))
	print('Recall Score: {}'.format(rf_recall))
	print('F-1 Score: {}'.format(rf_fbeta_score))

def main():
	"""
	Main method.
	"""
	train_split = 0.6
	tfidfClassification(train_split)
	Word2VecClassification(train_split)




if __name__ == '__main__':
	main()