import whoosh.index as index
from whoosh.qparser import QueryParser

import csv
import pandas as pd
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
ix = index.open_dir("index")

tfidf_df = pd.read_csv('./wine-reviews/winemag-tfidf-matrix.csv')
tf_idf_doc_dict = tfidf_df.to_dict()

def get_wines(keywords):
	
	results = {}
	N = len(tfidf_df.index)
	# Compute results
	for keyword in keywords.split():
		if not tf_idf_doc_dict.has_key(keyword):
			continue
		for i in range(0, N):
			if results.has_key(str(i)):
				results[str(i)] = results[str(i)] + float(tf_idf_doc_dict[keyword][i])
			else:
				results[str(i)] = 0 + float(tf_idf_doc_dict[keyword][i])
	
	# Remove results with zero score
	for key, value in results.items():
		if value == 0:
			del results[key]
		else:
			value = value/len(keywords)

	count = 10
	documents = []
	# Retrieve and print result document titles
	print "ID | Title | Score "
	for key, value in sorted(results.iteritems(), key=lambda (k,v): (v,k), reverse=True):
		with ix.searcher() as searcher:
			# Search for document by path index
			query = QueryParser("path", ix.schema).parse("/{0}".format(key))
			document = searcher.search(query)[0]
			documents.append({'title': document['title'], 'content': ""})
			print("{0} | {1} | {2} %".format(key, document['title'], round(float(results[key])*100.0, 2)))
		count = count - 1
		if count == 0:
			break

	return documents

if __name__ == "__main__":
	# Request input of keywords
	keywords = str(raw_input("Enter keywords to search, separated by spaces [Ctrl + C to quit] : "))
	keywords = " ".join([lemmatizer.lemmatize(word) for word in keywords.split()])
	get_wines(keywords)