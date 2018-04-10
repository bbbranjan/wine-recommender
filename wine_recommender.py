import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import time

stops = set(stopwords.words("english")) # Obtains common stop words from nltk
lemmatizer = WordNetLemmatizer()
cleaned_dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews_light.csv')
original_dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews.csv')

def get_wines(keywords):
	print(keywords)
	start_feature_time = time.time()
	vectorizer = TfidfVectorizer()
	tfidf_matrix = vectorizer.fit_transform(cleaned_dataframe['review'])
	feature_names = vectorizer.get_feature_names()
	
	query = keywords.split(' ')
	result_matrix = csr_matrix((tfidf_matrix.get_shape()[0],1), dtype=np.float64)
	
	start_result_matrix_time = time.time()

	query_matrix = vectorizer.transform(query)

	result_matrix = query_matrix*tfidf_matrix.T

	print(tfidf_matrix.shape)
	print(result_matrix.shape)

	result_array = result_matrix.sum(axis=0)
	# print(result_array.A)
	result_indices = np.argsort(result_array).A.tolist()[0][-10:]
	print(result_indices)

	print('Results obtained in: {}'.format(time.time() - start_feature_time))
	result_documents = []
	start_document_indexing_time = time.time()
	for index in list(reversed(result_indices)):
		document = cleaned_dataframe.iloc[index]
		original_document = original_dataframe.iloc[index]
		result_documents.append({'title': document['title'].title(), 'content': original_document['review']})
		print("| {0} | {1} | {2} | %".format(index, document['title'].title(), round(float(result_array.item(index))*100.0, 2)))

	return result_documents
	

def get_countries():
	country_list = list(cleaned_dataframe['country'])
	country_list = [c.strip() for c in country_list]
	country_list = list(set(country_list))
	country_list.sort()
	return country_list


if __name__ == "__main__":
	# Request input of keywords
	while(True):
		keywords = str(raw_input("Enter keywords to search, separated by spaces [Ctrl + C to quit] : "))
		keywords = " ".join([lemmatizer.lemmatize(word) for word in keywords.split()])
		keywords = " ".join([word for word in keywords.split() if word not in stops])
		get_wines(keywords)