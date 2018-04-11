import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import time

nltk.download('wordnet')
nltk.download('stopwords')

stops = set(stopwords.words("english")) # Obtains common stop words from nltk
lemmatizer = WordNetLemmatizer()
unfiltered_dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews_light.csv')
cleaned_dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews_light.csv')
original_dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews.csv')
filtered_cleaned_dataframe = cleaned_dataframe
filtered_original_dataframe = original_dataframe

# Gets most relevant documents
def get_wines(keywords):
	print(keywords) # Debug
	start_feature_time = time.time() # Times feature generation
	vectorizer = TfidfVectorizer()
	tfidf_matrix = vectorizer.fit_transform(filtered_cleaned_dataframe['review']) # Generate Tf-idf matrix
	feature_names = vectorizer.get_feature_names() 
	
	query = keywords.split(' ')
	result_matrix = csr_matrix((tfidf_matrix.get_shape()[0],1), dtype=np.float64)
	
	start_result_matrix_time = time.time()

	query_matrix = vectorizer.transform(query)

	result_matrix = query_matrix*tfidf_matrix.T # Get result matrix by multiplying query matrix with document matrix

	print(tfidf_matrix.shape)
	print(result_matrix.shape)

	result_array = result_matrix.sum(axis=0) # Sum up values of all results
	# print(result_array.A)
	result_indices = np.argsort(result_array).A.tolist()[0][-10:]
	print(result_indices)

	print('Results obtained in: {}'.format(time.time() - start_feature_time))
	result_documents = []
	start_document_indexing_time = time.time() # Time document retrieval
	for index in list(reversed(result_indices)):
		document = filtered_cleaned_dataframe.iloc[index]
		original_document = filtered_original_dataframe.iloc[index]
		result_documents.append({'title': document['title'].title(), 'content': original_document['review'], 'rating': original_document['rating'], 'price': original_document['price'], 'country': original_document['country']})
		print("| {0} | {1} | {2} | %".format(index, document['title'].title(), round(float(result_array.item(index))*100.0, 2)))

	return result_documents
	
# Gets list of unique countries in dataset
def get_countries():
	country_list = list(cleaned_dataframe['country'])
	country_list = [c.strip() for c in country_list]
	country_list = list(set(country_list))
	country_list.sort()
	return country_list

# Sets filters from UI
def set_filters(filter_dict):
	print(filter_dict)
	global filtered_cleaned_dataframe
	filtered_cleaned_dataframe = cleaned_dataframe
	if filter_dict.has_key('location[]'): # Gets rows of location(s) specified
		filtered_cleaned_dataframe = cleaned_dataframe.loc[cleaned_dataframe['country'].str.strip().isin(filter_dict['location[]'])]
	print(len(filtered_cleaned_dataframe.index))

	# Remove rows not falling in price range
	filtered_cleaned_dataframe = filtered_cleaned_dataframe.drop(filtered_cleaned_dataframe[filtered_cleaned_dataframe.price < float(filter_dict['price_range[]'][0])].index)
	filtered_cleaned_dataframe = filtered_cleaned_dataframe.drop(filtered_cleaned_dataframe[filtered_cleaned_dataframe.price > float(filter_dict['price_range[]'][1])].index)
	
	# Remove rows below minimum rating
	filtered_cleaned_dataframe = filtered_cleaned_dataframe.drop(filtered_cleaned_dataframe[filtered_cleaned_dataframe.rating < int(filter_dict['rating'][0])].index)
	
	# Filter out sentiment rows
	if filter_dict['sentiment'][0] == 'pos':
		filtered_cleaned_dataframe = filtered_cleaned_dataframe.drop(filtered_cleaned_dataframe[filtered_cleaned_dataframe.sentiment.str.strip() == 'neg'].index)
	elif filter_dict['sentiment'][0] == 'neg':
		filtered_cleaned_dataframe = filtered_cleaned_dataframe.drop(filtered_cleaned_dataframe[filtered_cleaned_dataframe.sentiment.str.strip() == 'pos'].index)
	else:
		pass

	print(len(filtered_cleaned_dataframe.index))

	global filtered_original_dataframe
	filtered_original_dataframe = original_dataframe.loc[filtered_cleaned_dataframe.index.values]
	return True


if __name__ == "__main__":
	# Request input of keywords
	while(True):
		keywords = str(raw_input("Enter keywords to search, separated by spaces [Ctrl + C to quit] : "))
		keywords = " ".join([lemmatizer.lemmatize(word) for word in keywords.split()])
		keywords = " ".join([word for word in keywords.split() if word not in stops])
		get_wines(keywords)