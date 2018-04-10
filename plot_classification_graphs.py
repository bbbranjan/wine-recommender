from gensim.models import word2vec

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import re
import string
import time

import matplotlib.pyplot as plt

dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews_with_sentiment.csv')

def plot_price_rating_sentiment():
	prices = list(dataframe['price'])
	ratings = list(dataframe['rating'])
	sentiments = list(dataframe['sentiment'])
	color = ['g' if s == 'pos' else 'r' for s in sentiments]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	plt.scatter(prices, ratings, c=color)
	plt.axis([0, 1200, 79, 101])
	# Move left y-axis and bottim x-axis to centre, passing through (0,0)
	ax.spines['left'].set_position('center')
	ax.spines['bottom'].set_position('center')

	# Eliminate upper and right axes
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')

	ax.set_xlabel('Price (USD)')
	ax.xaxis.set_label_coords(1.05, 0.45)
	ax.set_ylabel('Rating (%)').set_rotation(0)
	ax.yaxis.set_label_coords(0.58, 0.0)
	plt.title('Plot showing distribution of review sentiments in rating vs price')
	plt.legend(['positive', 'negative'])
	fig.tight_layout()
	plt.show()

def plot_location_sentiment():
	country_list = list(dataframe['country'])
	country_list = [c.strip() for c in country_list]
	country_list = list(set(country_list))
	sentiments = list(dataframe['sentiment'])
	positives = {}
	negatives = {}
	for country in country_list:
		positives[country] = 0
		negatives[country] = 0

	for index, row in dataframe.iterrows():
		if row['sentiment'] == 'pos':
			positives[row['country'].strip()] += 1
		else:
			negatives[row['country'].strip()] += 1

	for country in country_list:
		if positives[country] < 10 or negatives[country] == 10:
			del positives[country]
			del negatives[country]
			country_list.remove(country)		

	ind = np.arange(len(country_list))
	width = 0.35

	print(len(country_list), len(positives.values()), len(negatives.values()))

	p1 = plt.bar(ind, positives.values(), width)
	p2 = plt.bar(ind, negatives.values(), width, bottom=positives.values())

	plt.ylabel('Counts')
	plt.title('Sentiment of wine reviews by country')

	plt.xticks(ind, country_list, rotation='vertical')
	plt.yticks(np.arange(0, max(positives.values())+1, 1000))
	plt.legend((p1[0], p2[0]), ('Positive', 'Negative'))

	plt.show()

def plot_variety_price():
	varieties = list(dataframe['variety'])
	varieties = [v.strip() for v in varieties]
	varieties = list(set(varieties))

	prices = {}

	for variety in varieties:
		prices[variety] = 0.0

	for index, row in dataframe.iterrows():
		if row['price'] > prices[row['variety'].strip()]:
			prices[row['variety'].strip()] = float(row['price'])

	print(len(varieties))
	# print(prices)

	price_list = list(prices.values())
	price_list.sort(reverse=True)

	plt.plot(price_list, 'r--')
	plt.ylabel('Price (USD)')
	plt.xlabel('Wine Varieties (unlabelled)')
	plt.title('Price distribution of most expensive wines by variety')
	plt.show()

def plot_location_price():
	country_list = list(dataframe['country'])
	country_list = [c.strip() for c in country_list]
	country_list = list(set(country_list))
	country_list = [c[:8] for c in country_list]
	prices = {}

	for country in country_list:
		prices[country] = 0.0

	for index, row in dataframe.iterrows():
		if float(row['price']) > prices[row['country'].strip()[:8]]:
			prices[row['country'].strip()[:8]] = float(row['price'])

	print(prices)
	ind = np.arange(len(country_list))
	plt.bar(ind, prices.values())

	plt.ylabel('Price (USD)')
	plt.xlabel('Country')
	plt.xticks(ind, country_list, rotation='vertical')
	plt.title('Bar plot of maximum wine price by country')
	plt.show()

def plot_data_point_graph():
	dataframe = pd.read_csv('./wine-reviews/cleaned_wine_reviews_with_sentiment.csv')
	neg_rows = int(0.1*len(dataframe.index)/2)
	pos_rows = int(0.1*len(dataframe.index)/2)
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

	pca = PCA(n_components=2)
	vectorizer = TfidfVectorizer(analyzer='word', strip_accents='ascii', lowercase=True, max_features=10000)
	features = vectorizer.fit_transform(dataframe['review'])
	features_nd = features.toarray() # for easy usage

	decomposed_features = pca.fit_transform(features_nd)
	print(len(decomposed_features))
	x_features = []
	y_features = []
	sentiments = list(dataframe['sentiment'])
	for i in range(0, len(decomposed_features)):
		x_features.append(decomposed_features[i][0])
		y_features.append(decomposed_features[i][1])

	colors = ['g' if s == 'pos' else 'r' for s in sentiments]
	plt.scatter(x_features, y_features, c=colors)

	plt.show()

plot_location_price()