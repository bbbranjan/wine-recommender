# Create your views here.
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, render_to_response

from django.http import Http404, HttpResponse, HttpResponseRedirect , JsonResponse
from django.views import View

from wine_recommender import get_wines, get_countries
from autocorrect import spell
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from query_popularity import get_popular_terms

import time

stops = set(stopwords.words("english")) # Obtains common stop words from nltk
lemmatizer = WordNetLemmatizer()

class searchView(View):

    # Handle POST requests
    def post(self, request):

        countries = get_countries()

        # Start timer
        start_time = time.time()

        query = ' '.join([spell(word) for word in request.POST['term'].split()])

        query = " ".join([lemmatizer.lemmatize(word) for word in query.split()])
        query = " ".join([word for word in query.split() if word not in stops])

        # Debug request POST
        print(request.POST)
        
        # Get recommended wines 
        result = get_wines(query)

        time_taken = time.time() - start_time

        print(time_taken)

        
        # Render frontend based on results obtained
        return render(request,'search.html',{'query': query, 'result': result,'locations':countries, 'popular_terms': sorted(get_popular_terms(query).iteritems(), key=lambda (k,v): (v,k), reverse=True)})
    
    # Handle GET requests
    def get(self, request):
        
        return render(request,'search.html',{'popular_terms': sorted(get_popular_terms().iteritems(), key=lambda (k,v): (v,k), reverse=True), 'locations':get_countries()})

class filterView(View):

    # Handle POST requests
    def post(self, request):
        
        # Debug request POST
        print(request.POST)

        location_list = [countries[i] for i in request.POST.getlist('location')]

        print(location_list)

        # Render frontend based on results obtained
        return JsonResponse({'result':True})

