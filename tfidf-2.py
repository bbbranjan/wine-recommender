import csv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cleaned_dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews_light.csv')
original_dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews.csv')
original_dataframe = original_dataframe.set_index(['title'])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_dataframe['review'])
feature_names = vectorizer.get_feature_names()

query = 'fruit wine cranberry'.split()
result_matrix = csr_matrix((tfidf_matrix.get_shape()[0],1), dtype=np.float64)
# print(type(result_matrix))
for term in query:
	for i in range(0, result_matrix.shape[0]):
		result_matrix[i,0] += tfidf_matrix.getrow(i).getcol(feature_names.index(term)).max()

result_scores = result_matrix.getcol(0).toarray()
result_scores = np.asarray(map(lambda x: x[0], result_scores))
# print(type(result_scores), len(result_scores))
result_indices = np.argsort(result_scores)[-10:]
result_documents = []

for index in result_indices:
	document = cleaned_dataframe.iloc[index]
	original_document = original_dataframe.loc[document['title']]
	result_documents.append({'title': document['title'], 'content': original_document.loc['review']})

print result_documents
