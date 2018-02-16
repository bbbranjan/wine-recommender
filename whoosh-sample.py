import os

from whoosh.index import create_in
from whoosh.fields import *

import csv

schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
if not os.path.exists("index"):
    os.mkdir("index")
ix = create_in("index", schema)
writer = ix.writer()
with open('./wine-reviews/winemag-data_first10.csv', 'rb') as csvfile:
	review_reader = csv.DictReader(csvfile)
	count = 0;
	for row in review_reader:
		writer.add_document(title=unicode("{0}, {1}, {2}, {3}".format(row['variety'], row['designation'], row['region_1'], row['country']), errors='ignore'), path=unicode("/{0}".format(count), errors='ignore'), content=unicode("{0}".format(row['description']), errors='ignore'))
		count = count + 1
writer.commit()
# writer.add_document(title=u"First document", path=u"/a", content=u"This is the first document we've added!")
# writer.add_document(title=u"Second document", path=u"/b", content=u"The second one is even more interesting!")

from whoosh.qparser import QueryParser
with ix.searcher() as searcher:
	query = QueryParser("content", ix.schema).parse("gold")
	results = searcher.search(query)
	for result in results:
		print(result['title'])