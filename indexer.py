import os

from whoosh.index import create_in
from whoosh.fields import *

import math
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import csv

stops = set(stopwords.words("english")) # Obtains common stop words from nltk
lemmatizer = WordNetLemmatizer()

def remove_punctuation(x):
    # Lowercases all words
    x = x.lower()
    # Removes non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removes (replaces with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)

def remove_numbers(x):
	# Removes numbers
	return re.sub(r'[0-9]+', r'', x)

def remove_stopwords(x):
    # Removes all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)

def lemmatize_words(x):
    # Lemmatizes words
    lemmatizedWords = [lemmatizer.lemmatize(word) for word in x.split()]
    return " ".join(lemmatizedWords)

def main():
	
	# Create index
	schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
	if not os.path.exists("index"):
	    os.mkdir("index")
	ix = create_in("index", schema)
	writer = ix.writer()

	# Load the dataframe
	dataframe = pd.read_csv('./wine-reviews/winemag-data_first100.csv')
	# Remove punctuation
	dataframe["description"] = dataframe["description"].map(remove_punctuation)
	# Remove numbers
	dataframe["description"] = dataframe["description"].map(remove_numbers)  
	# Remove stopwords
	dataframe["description"] = dataframe["description"].map(remove_stopwords)
	# Lemmatize words
	dataframe["description"] = dataframe["description"].map(lemmatize_words)

	N = len(dataframe)
	dfs = {}
	idf = {}
	terms = []

	# Calculate inverted document frequency
	for i in range(0, N):
		for term in dataframe["description"].iloc[i].split():
			if term in terms:
				dfs[term] = dfs[term] + 1
			else:
				terms.append(term)
				dfs[term] = 1
			idf[term] = math.log10(N/float(dfs[term]))


	tf_doc_dict = {}
	tf_dict = {}

	# Calculate term frequency
	for term in terms:
		tf_dict = {}
		for i in range(0, N):
			tf = 0
			for word in dataframe["description"].iloc[i].split():
				if word == term:
					tf = tf + 1
			if tf > 0:
				# print "{0} {1}".format(i, term)
				tf_dict[str(i)] = (1 + math.log10(tf))
			else:
				tf_dict[str(i)] = 0
		tf_doc_dict[term] = tf_dict

	# Calculate tf-idf dictionary
	tf_idf_dict = {}
	tf_idf_doc_dict = {}
	for term in terms:
		tf_idf_dict = {}
		for i in range(0, N):
			tfdict = tf_doc_dict[term]
			tf_idf_dict[str(i)] = float(tfdict[str(i)]) * float(idf[term])
			# tf_idf_dict[str(i)] = round(tf_idf_dict[str(i)], 2)
		tf_idf_doc_dict[term] = tf_idf_dict

	# Store tf-idf dictionary in dataframe and save as csv file
	tfidf_df = pd.DataFrame(data=tf_idf_doc_dict)

	tfidf_df.to_csv('./wine-reviews/winemag-tfidf-matrix.csv', index=True)

	print "Indexer is being run"

	# Store tokens in light CSV file
	dataframe.to_csv('./wine-reviews/winemag-data_first100_light.csv', index=False)

	with open('./wine-reviews/winemag-data_first100_light.csv', 'rb') as csvfile:
	    # Read the csv file as a dict
		review_reader = csv.DictReader(csvfile)
		id_num = 0; # Track row id
		for row in review_reader:
	        # Add the processed description as a document to the index
			writer.add_document(title=unicode("{0}, {1}, {2}, {3}".format(row['variety'], row['designation'], row['region_1'], row['country']), errors='ignore'), path=unicode("/{0}".format(id_num), errors='ignore'), content=unicode("{0}".format(row['description']), errors='ignore'))
			id_num = id_num + 1

	# Commit added documents to index
	writer.commit()


if __name__ == "__main__":
    main()