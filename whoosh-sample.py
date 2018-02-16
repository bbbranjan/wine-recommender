import os

from whoosh.index import create_in
from whoosh.fields import *

import pandas as pd
from nltk.corpus import stopwords
import re
import string
import csv
import nltk

stops = set(stopwords.words("english")) # Obtains common stop words from nltk

def removePunctuation(x):
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)

def removeStopwords(x):
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)

# Creating the index
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
if not os.path.exists("index"):
    os.mkdir("index")
ix = create_in("index", schema)
writer = ix.writer()

# Loading the dataframe
dataframe = pd.read_csv('./wine-reviews/winemag-data_first10.csv')
# Removing punctuation
dataframe["description"] = dataframe["description"].map(removePunctuation) 
# Removing stopwords
dataframe["description"] = dataframe["description"].map(removeStopwords) 

dataframe.to_csv('./wine-reviews/winemag-data_first10_light.csv', index=False)

with open('./wine-reviews/winemag-data_first10_light.csv', 'rb') as csvfile:
    # Reads the csv file as a dict
	review_reader = csv.DictReader(csvfile)
	id_num = 0; # Tracks row id
	for row in review_reader:
        # Adding the processed description as a document to the index
		writer.add_document(title=unicode("{0}, {1}, {2}, {3}".format(row['variety'], row['designation'], row['region_1'], row['country']), errors='ignore'), path=unicode("/{0}".format(id_num), errors='ignore'), content=unicode("{0}".format(row['description']), errors='ignore'))
		id_num = id_num + 1

# Committing added documents to index
writer.commit()
# writer.add_document(title=u"First document", path=u"/a", content=u"This is the first document we've added!")

from whoosh.qparser import QueryParser
# Requesting input of keywords
keywords = str(raw_input("Enter keywords to search, separated by spaces: "))
with ix.searcher() as searcher:
    # Parsing index for keywords
	query = QueryParser("content", ix.schema).parse(keywords)
    # Executing search query
	results = searcher.search(query)
	for result in results:
        # Printing all results
		print(result['title'])