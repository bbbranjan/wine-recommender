import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stops = set(stopwords.words("english")) # Obtains common stop words from nltk
lemmatizer = WordNetLemmatizer()
cleaned_dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews_light.csv')
original_dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews.csv')

def get_wines(keywords):
	print(keywords)
	vectorizer = TfidfVectorizer()
	tfidf_matrix = vectorizer.fit_transform(cleaned_dataframe['review'])
	feature_names = vectorizer.get_feature_names()

	query = keywords.split()
	result_matrix = csr_matrix((tfidf_matrix.get_shape()[0],1), dtype=np.float64)
	
	for term in query:
		for i in range(0, result_matrix.shape[0]):
			if term in feature_names:
				result_matrix[i,0] += tfidf_matrix.getrow(i).getcol(feature_names.index(term)).max()

	result_scores = result_matrix.getcol(0).toarray()
	result_scores = np.asarray(map(lambda x: x[0], result_scores))
	
	result_indices = np.argsort(result_scores)[-10:]
	result_documents = []

	for index in list(reversed(result_indices)):
		document = cleaned_dataframe.iloc[index]
		original_document = original_dataframe.iloc[index]
		result_documents.append({'title': document['title'], 'content': original_document['review']})
		print("| {0} | {1} | {2} | %".format(index, document['title'], round(float(result_scores.item(index))*100.0, 2)))

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